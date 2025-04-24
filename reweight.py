import subprocess
import io
import sys
from collections import Counter
import re
import os, tempfile, shutil
import re, json

# Can install these via apt:
import pandas as pd
import numpy as np

import argparse
help_msg = "Calculate Ceph OSD reweights using deviation from mean utilisation %."
help_msg += " Only calculate - reweight is still a manual task for you to review."
help_msg += " Accounts for any PGs currently being remapped (ceph pg dump pgs_brief), by analysing the utilisation after current remap completes."
parser = argparse.ArgumentParser(description=help_msg)
parser.add_argument('-p', '--pool', type=str, required=True, help="Focus on this Ceph pool")
parser.add_argument('-m', '--min', type=float, default=5.0, help='Deviation threshold. E.g. 5 means: ignore OSDs within mean util %% +-5%%')
parser.add_argument('-l', '--limit', type=int, default=None, help='Optional: limit to N OSDs with biggest deviation')
parser.add_argument('-o', '--osd', type=int, default=None, help='Optional: print detailed information for this OSD number')
parser.add_argument('-s', '--cephadm', action='store_true', help='Run Ceph query commands via cephadm shell')
parser.add_argument('-e', '--exclude-host', action='append', default=[], help='Exclude these hosts matching these regex patterns. Can be used multiple times.')
args = parser.parse_args()
if args.min < 0 or args.min > 100:
    raise ValueError(f'Argument "min" must be between 0 and 100, but provided value is outside range: {args.limit}')
args.min *= 0.01  # convert to 0->1
if args.limit is not None and args.limit < 1:
    raise ValueError(f'Argument "limit" if set must be 1 or greater, not {args.limit}')

pool_osds_cmd = f'ceph pg ls-by-pool {args.pool} | tr -s " " | cut -d" " -f15 | tail -n+2'
util_cmd = "ceph osd df plain | grep up | tr -s ' ' | sed 's/^ //' | cut -d' ' -f1,4,17"
util_cols = ['id', 'wgt', 'util']
pgs_cmd = "ceph pg dump pgs_brief"
draining_cmd = "ceph orch osd rm status | tail -n+2 | cut -d' ' -f1"
osd2host_cmd = 'ceph osd metadata -f json | jq -r \'.[] | "\\(.id) \\(.hostname)"\''
osd2host_cols = ['id', 'host']

if args.cephadm:
    mount_dir = tempfile.mkdtemp(prefix="cephadm_", dir="/tmp")
    container_mount_dir = f"/mnt/{os.path.basename(mount_dir)}"

    commands_script = f"""#!/bin/bash
    {pool_osds_cmd} > {container_mount_dir}/pool_osds.out
    {util_cmd} > {container_mount_dir}/util.out
    {draining_cmd} > {container_mount_dir}/draining.out
    {pgs_cmd} > {container_mount_dir}/pgs.out
    {osd2host_cmd} > {container_mount_dir}/osd2host.out
    """

    commands_script_path = os.path.join(mount_dir, "commands.sh")
    with open(commands_script_path, "w") as f:
        f.write(commands_script)
    os.chmod(commands_script_path, 0o755)

    composite_cmd = f"./cephadm shell --mount {mount_dir} -- bash {container_mount_dir}/commands.sh"
    subprocess.run(composite_cmd, shell=True, check=True)

    with open(os.path.join(mount_dir, "pool_osds.out"), "r") as f:
         pool_osds_stdout = f.read()
    with open(os.path.join(mount_dir, "util.out"), "r") as f:
         util_stdout = f.read()
    with open(os.path.join(mount_dir, "draining.out"), "r") as f:
         draining_stdout = f.read()
    with open(os.path.join(mount_dir, "pgs.out"), "r") as f:
         pgs_stdout = f.read()
    with open(os.path.join(mount_dir, "osd2host.out"), "r") as f:
         osd2host_stdout = f.read()
else:
    pass

if not args.cephadm:
    print(f"Running: {pool_osds_cmd}", file=sys.stderr)
    pool_osds_stdout = subprocess.check_output(pool_osds_cmd, shell=True, text=True)
pool_osds = set()
for line in re.findall(r'\[[\d,]+\]', pool_osds_stdout):
   nums = line.strip('[]').split(',')
   pool_osds.update(int(n) for n in nums)
if len(pool_osds) == 0:
    msg = "Ceph command failed."
    if not args.cephadm:
        msg += " Does pool exist? Does it need to be run via a cephadm shell?"
    msg += "\n    " + pool_osds_cmd
    raise Exception(msg)

if not args.cephadm:
    print(f"Running: {util_cmd}", file=sys.stderr)
    proc = subprocess.run([util_cmd], shell=True, capture_output=True, text=True)
    util_stdout = proc.stdout

df_util = pd.read_csv(io.StringIO(util_stdout), sep=r'\s+', header=None, names=util_cols)
df_util['id'] = df_util['id'].astype(int)
df_util = df_util.set_index('id').sort_index()
df_util = df_util[df_util.index.isin(pool_osds)]
# Convert util to same 0->1 range as weight, avoids confusion.
df_util['util'] *= 0.01

if not args.cephadm:
    print(f"Running: {draining_cmd}", file=sys.stderr)
    proc = subprocess.run([draining_cmd], shell=True, capture_output=True, text=True)
    draining_stdout = proc.stdout

df_draining = pd.read_csv(io.StringIO(draining_stdout), sep=r'\s+', header=None, names=['id'])
if not df_draining.empty:
    df_draining['id'] = df_draining['id'].astype(int)
    print(f"Ignoring draining OSDs: {df_draining['id'].to_numpy()}", file=sys.stderr)
    df_util = df_util[~df_util.index.isin(df_draining['id'])]

if not args.cephadm:
    print(f"Running: {pgs_cmd}", file=sys.stderr)
    proc = subprocess.run([pgs_cmd], shell=True, capture_output=True, text=True)
    pgs_stdout = proc.stdout
df_pgs = pd.read_csv(io.StringIO(pgs_stdout), sep=r'\s+', header=0)
df_pgs = df_pgs.set_index('PG_STAT')
def parse_osd_list(value):
    clean_str = value.strip('[]')
    return {int(x.strip()) for x in clean_str.split(',') if x.strip()}
current = Counter()
changes_active = Counter()
changes_waiting = Counter()
backfilling_pgs_new_ups = {}
up_osds_all = set()
for _, row in df_pgs.iterrows():
    up_osds = parse_osd_list(row['UP'])
    up_osds_all.update(up_osds)
    acting_osds = parse_osd_list(row['ACTING'])
    waiting = 'wait' in row['STATE']
    backfilling = 'backfilling' in row['STATE']
    if backfilling:
        pg_id = row.name
        backfilling_pgs_new_ups[pg_id] = []
    for osd in acting_osds:
        current[osd] += 1
    for osd in acting_osds - up_osds:
        if waiting:
            changes_waiting[osd] -= 1
        else:
            changes_active[osd] -= 1
    for osd in up_osds - acting_osds:
        if waiting:
            changes_waiting[osd] += 1
        else:
            if backfilling:
                backfilling_pgs_new_ups[pg_id].append(osd)
                # Will calculate the percent transferred later
            else:
                changes_active[osd] += 1

# Backfilling PG stats
backfilling_osds = set()
if backfilling_pgs_new_ups:
    bash_script = "#!/bin/bash\n"
    bash_script += "echo \"[\"\n"
    bash_script += "first=1\n"
    for pg, allowed_peers in backfilling_pgs_new_ups.items():
        # Convert allowed peers to strings since JSON output uses strings
        allowed_list = [str(peer) for peer in allowed_peers]
        allowed_json = json.dumps(allowed_list)
        # Fetch the PG query output and determine total objects.
        bash_script += f"query=$(ceph pg {pg} query -f json)\n"
        bash_script += "total=$(echo \"$query\" | jq '.info.stats.stat_sum.num_objects')\n"
        # Use jq to filter peer info based on allowed peers; note the careful quoting.
        bash_script += "json_peer=$(echo \"$query\" | jq --argjson allowed '" + allowed_json + "' --arg total \"$total\" '[.peer_info[] | select(.peer as $p | $allowed | index($p)) | {peer: .peer, percent_missing: ((.stats.stat_sum.num_objects_missing/($total|tonumber))), percent_misplaced: ((.stats.stat_sum.num_objects_misplaced/($total|tonumber)))}]')\n"
        # Build a JSON object for this PG.
        bash_script += f"json_pg=$(printf '{{\"pg\": \"{pg}\", \"total_objects\": %s, \"peers\": %s}}' \"$total\" \"$json_peer\")\n"
        # Output with a comma prefix if not the first element.
        bash_script += "if [ $first -eq 1 ]; then echo \"$json_pg\"; first=0; else echo \",\"; echo \"$json_pg\"; fi\n"
    bash_script += "echo \"]\"\n"

    # Determine where to write the script based on cephadm mode
    if args.cephadm:
        # Use the previously created mount_dir and container_mount_dir
        script_path = os.path.join(mount_dir, "backfill_stats.sh")
        container_script_path = os.path.join(container_mount_dir, "backfill_stats.sh")
    else:
        script_path = tempfile.mktemp(prefix="backfill_stats_", dir="/tmp")
        container_script_path = script_path

    with open(script_path, "w") as f:
        f.write(bash_script)
    os.chmod(script_path, 0o755)

    # Run the script and capture its output
    print(f"Querying Ceph for PGs backfill progress ...", file=sys.stderr)
    if args.cephadm:
        composite_backfill_cmd = f"./cephadm shell --mount {mount_dir} -- bash {container_script_path}"
        result = subprocess.run(composite_backfill_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    else:
        result = subprocess.run(script_path, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    backfill_output = result.stdout
    bf_stats = json.loads(backfill_output)
    bf_stats = {x['pg']:x for x in bf_stats}
    for pg in bf_stats.keys():
        pg_stats = bf_stats[pg]
        for peer_stats in pg_stats['peers']:
            osd = int(peer_stats['peer'])
            if osd in up_osds_all:
                pct = peer_stats['percent_missing']
                changes_active[osd] += (1-pct)
                changes_waiting[osd] += pct

if args.cephadm:
    shutil.rmtree(mount_dir)

rows = []
for osd in current:
    pgs_curr = current[osd]
    pgs_change_active = round(changes_active[osd], 1)
    pgs_change_wait = round(changes_waiting[osd], 1)
    row_data = {'OSD': int(osd), 'PGs current': pgs_curr, 'Remap now': pgs_change_active, 'Remap waiting': pgs_change_wait}
    rows.append(row_data)
missing_osds = sorted(list(set(df_util.index) - set(current.keys())))
for osd in missing_osds:
    # these are new osds
    pgs_curr = 0
    pgs_change_active = 0
    pgs_change_wait = changes_waiting[osd]
    row_data = {'OSD': int(osd), 'PGs current': pgs_curr, 'Remap now': pgs_change_active, 'Remap waiting': pgs_change_wait}
    rows.append(row_data)
df_remap = pd.DataFrame(rows).set_index('OSD')
df_remap['PGs up'] = (df_remap['PGs current'] + df_remap['Remap now'] + df_remap['Remap waiting']).round(0).astype('int')
df_util = df_util.join(df_remap[['PGs current', 'Remap now', 'Remap waiting', 'PGs up']])
df_util['util curr'] = df_util['util'].round(4)
util_change = (df_util['Remap now'] + df_util['Remap waiting']) / df_util['PGs current']
df_util['util up'] = (df_util['util curr'] * (1 + util_change)).round(4)
fna = (df_util['PGs current']==0).to_numpy()
if fna.any():
    # For new OSDs that haven't receiving a full PG yet, assume util = pool average
    df_util.loc[fna, 'util up'] = df_util['util up'][~fna].mean()
df_util = df_util.drop('util', axis=1)

if args.osd is not None:
    if args.osd not in df_util.index:
        print(f'Warning: specified osd "{args.osd}" is not in utilisation data.', file=sys.stderr)
    else:
        print(df_util.loc[args.osd], file=sys.stderr)

if not args.cephadm:
    print(f"Running: {osd2host_cmd}", file=sys.stderr)
    proc = subprocess.run([osd2host_cmd], shell=True, capture_output=True, text=True)
    osd2host_stdout = proc.stdout

df_osd2host = pd.read_csv(io.StringIO(osd2host_stdout), sep=r'\s+', header=None, names=osd2host_cols)
df_osd2host = df_osd2host.set_index('id')
df_util = df_util.join(df_osd2host)

# Exclude specified hosts based on regex patterns
if args.exclude_host:
    f_exclude = np.zeros(len(df_util), dtype=bool)
    for pat in args.exclude_host:
        # Rewrite capture groups to non-capturing to avoid warning
        pat = re.sub(r'\((?!\?P<|\?[:=!])', '(?:', pat)
        f_exclude |= df_util['host'].str.contains(pat, regex=True)
    if f_exclude.any():
        excluded_hosts = df_util['host'][f_exclude].unique()
        print(f"Excluding hosts: {list(excluded_hosts)}", file=sys.stderr)
        df_util = df_util[~f_exclude]

# Calculate deviation from mean utilisation
mean_util = df_util['util up'].mean()
print(f"# mean_util = {100*mean_util:.1f}%", file=sys.stderr)
df_util['deviation'] = df_util['util up'] - mean_util
df_util['dev abs'] = df_util['deviation'].abs()
total_pgs = df_util['PGs up'].sum()

df_top = df_util.copy()

# Ignore small deviations
f_below_min_dev = df_top['dev abs']<args.min
if f_below_min_dev.any():
    print(f'Ignoring {np.sum(f_below_min_dev)} OSDs as their deviation below threshold', file=sys.stderr)
    df_top = df_top[~f_below_min_dev]
    if df_top.empty:
        print("No significant deviations", file=sys.stderr)
        quit()

# Discard OSDs with util < mean but weight=1.0, because no room to increase weight.
f = (df_top['wgt']==1.0) & (df_top['util up']<mean_util)
if f.any():
    if np.sum(f) < 5:
        maxxed_osds = sorted(df_top.index[f].tolist())
        msg = f"Ignoring {np.sum(f)} OSDs because their util is < mean but their weight=1.0, can't increase weight: {maxxed_osds}"
    else:
        msg = f"Ignoring {np.sum(f)} OSDs because their util is < mean but their weight=1.0, can't increase weight."
    print(msg, file=sys.stderr)
    df_top = df_top[~f]
    if df_top.empty:
        print("No significant deviations with weight<1", file=sys.stderr)
        quit()

# Restrict to top-N biggest deviations
if args.limit is not None:
    print(f"Only analysing the top {args.limit} OSDs with biggest deviations", file=sys.stderr)
    df_top = df_top.sort_values('dev abs', ascending=False).iloc[:args.limit]

# Set new weights to exactly where we want them, not a vague shift.
new_weight = df_top['wgt'] * (mean_util / df_top['util up'])
df_top['wgt shift'] = new_weight - df_top['wgt']

# Check if a shifted weight would be > 1.
df_top['new wgt'] = df_top['wgt'] + df_top['wgt shift']
new_weight_max = (df_top['wgt'] + df_top['wgt shift']).max()
new_weights = df_top[['host', 'util curr', 'util up', 'wgt', 'new wgt', 'PGs up']].copy()
if new_weight_max <= 1:
    # Simple, just change these selected OSD
    pass
else:
    # # Oh dear, need to change ALL weights.
    # # Because if we just downscale the top new weights, 
    # # then they can be almost identical to current weights.
    # scale_factor = 1 / new_weight_max
    # df_util_wo_top = df_util[~df_util.index.isin(df_top.index)].copy()           
    # df_util_wo_top['new wgt'] = df_util_wo_top['wgt'] * scale_factor
    # cols = ['host', 'util curr', 'util up', 'wgt', 'new wgt', 'PGs up']
    # new_weights1 = df_top[cols]
    # new_weights2 = df_util_wo_top[cols]
    # new_weights = pd.concat([new_weights1, new_weights2])
    
    # But that may put big load on Ceph, so just cap the max.
    # But at some point, must reweight all OSDs.
    f_above_1 = new_weights['new wgt'] > 1.0
    if f_above_1.any():
        print(f"Clipping {np.sum(f_above_1)} new weights to maximum 1.0", file=sys.stderr)
        new_weights['new wgt'] = new_weights['new wgt'].clip(upper=1.0)

new_weights['new wgt'] = new_weights['new wgt'].round(5)
new_weights['shift'] = (new_weights['new wgt'] - new_weights['wgt']).round(3)
new_weights = new_weights[new_weights['shift'] != 0]
if new_weights.empty:
    print("No significant reweights needed", file=sys.stderr)
    quit()
new_weights = new_weights.sort_values('util up', ascending=False)
new_weights['PGs mv'] = ((new_weights['shift'] / new_weights['wgt']) * new_weights['PGs up']).round(1)
cols_sorted = ['host', 'util curr', 'util up', 'wgt', 'new wgt', 'shift', 'PGs up', 'PGs mv']

print(f"# new_weights: count={len(new_weights)}", file=sys.stderr)
print(new_weights[cols_sorted], file=sys.stderr)

pgs_pushed = abs((new_weights['PGs mv'][new_weights['PGs mv']<0]).sum())
pgs_pulled = abs((new_weights['PGs mv'][new_weights['PGs mv']>0]).sum())
max_pgs_written = max(pgs_pushed, pgs_pulled)
print(f"# PGs to write = {max_pgs_written:.0f} = {100*max_pgs_written/total_pgs:.1f}%", file=sys.stderr)

# Write Bash commands to stdout
print("#!/bin/bash")
print("ceph osd set norebalance")
print("sleep 5")
for i in range(len(new_weights)):
    row = new_weights.iloc[i]
    osd_id = row.name
    if isinstance(osd_id, (int, np.int64)):
        osd_id = "osd."+str(osd_id)
    weight = row['new wgt']
    print(f"ceph osd reweight {osd_id} {weight:.5f}")
print('echo "To trigger rebalance run: ceph osd unset norebalance"')
print('echo "Or instead, re-run this script to refine weights."')
print("")

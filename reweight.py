import subprocess
import io
import os
import datetime as dt
from collections import Counter
import re

# Can install these via apt:
import pandas as pd
import numpy as np
import click

import argparse
help_msg = "Calculate Ceph OSD reweights using deviation from mean utilisation %."
help_msg += " Only calculate - reweight is still a manual task for you to review."
help_msg += " Accounts for any PGs currently being remapped (ceph pg dump pgs_brief), by analysing the utilisation after current remap completes."
parser = argparse.ArgumentParser(description=help_msg)
parser.add_argument('-p', '--pool', type=str, required=True, help="Focus on this Ceph pool")
parser.add_argument('-m', '--min', type=float, default=5.0, help='Deviation threshold. E.g. 5 means: ignore OSDs within mean util %% +-5%%')
parser.add_argument('-d', '--outdir', type=str, default="reweight-job-dir", help="Store reweight job script in this folder. This Python script does not actually apply weights")
parser.add_argument('-l', '--limit', type=int, default=None, help='Optional: limit to N OSDs with biggest deviation')
parser.add_argument('-o', '--osd', type=int, default=None, help='Optional: print detailed information for this OSD number')
parser.add_argument('-s', '--cephadm', action='store_true', help='Run Ceph query commands via cephadm shell')
args = parser.parse_args()
if args.min < 0 or args.min > 100:
    raise ValueError(f'Argument "min" must be between 0 and 100, but provided value is outside range: {args.limit}')
args.min *= 0.01  # convert to 0->1
if args.limit is not None and args.limit < 1:
    raise ValueError(f'Argument "limit" if set must be 1 or greater, not {args.limit}')

pool_osds_cmd = f'ceph pg ls-by-pool {args.pool} | tr -s " " | cut -d" " -f15 | tail -n+2'
util_cmd = "ceph osd df plain | grep up | tr -s ' ' | sed 's/^ //' | cut -d' ' -f1,4,17"
util_cols = ['id', 'weight', 'util']
pgs_cmd = "ceph pg dump pgs_brief"
draining_cmd = "ceph orch osd rm status | tail -n+2 | cut -d' ' -f1"

if args.cephadm:
    pool_osds_cmd = "./cephadm shell -- " + pool_osds_cmd
    util_cmd = "./cephadm shell -- " + util_cmd
    pgs_cmd = "./cephadm shell -- " + pgs_cmd
    draining_cmd = "./cephadm shell -- " + draining_cmd

print(f"Running: {pool_osds_cmd}")
output = subprocess.check_output(pool_osds_cmd, shell=True, text=True)
pool_osds = set()
for line in re.findall(r'\[[\d,]+\]', output):
   nums = line.strip('[]').split(',')
   pool_osds.update(int(n) for n in nums)
if len(pool_osds) == 0:
    msg = "Ceph command failed."
    if not args.cephadm:
        msg += " Does pool exist? Does it need to be run via a cephadm shell?"
    msg += "\n    " + pool_osds_cmd
    raise Exception(msg)

print(f"Running: {util_cmd}")
proc = subprocess.run([util_cmd], shell=True, capture_output=True, text=True)
df_util = pd.read_csv(io.StringIO(proc.stdout), sep=r'\s+', header=None, names=util_cols)
df_util['id'] = df_util['id'].astype(int)
df_util = df_util.set_index('id').sort_index()
df_util = df_util[df_util.index.isin(pool_osds)]
# Convert util to same 0->1 range as weight, avoids confusion.
df_util['util'] *= 0.01

# Ignore OSDs currently being drained
print(f"Running: {draining_cmd}")
proc = subprocess.run([draining_cmd], shell=True, capture_output=True, text=True)
df_draining = pd.read_csv(io.StringIO(proc.stdout), sep=r'\s+', header=None, names=['id'])
if not df_draining.empty:
    df_draining['id'] = df_draining['id'].astype(int)
    print(f"Ignoring draining OSDs: {df_draining['id'].to_numpy()}")
    df_util = df_util[~df_util.index.isin(df_draining['id'])]

# Suppose a remap is already underway. Then instead of using current utilisation, 
# calculate and use the post-remap utilisation.
print(f"Running: {pgs_cmd}")
proc = subprocess.run([pgs_cmd], shell=True, capture_output=True, text=True)
df_pgs = pd.read_csv(io.StringIO(proc.stdout), sep=r'\s+', header=0)
df_pgs = df_pgs.set_index('PG_STAT')
def parse_osd_list(value):
    clean_str = value.strip('[]')
    return {int(x.strip()) for x in clean_str.split(',') if x.strip()}
current = Counter()
changes_active = Counter()
changes_waiting = Counter()
for _, row in df_pgs.iterrows():
    up_osds = parse_osd_list(row['UP'])
    acting_osds = parse_osd_list(row['ACTING'])
    waiting = 'wait' in row['STATE']
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
            changes_active[osd] += 1
rows = []
for osd in current:
    pgs_curr = current[osd]
    pgs_change_active = changes_active[osd]
    pgs_change_wait = changes_waiting[osd]
    row_data = {'OSD': int(osd), 'PGs current': pgs_curr, 'Remap now': pgs_change_active, 'Remap waiting': pgs_change_wait}
    rows.append(row_data)
df_remap = pd.DataFrame(rows).set_index('OSD')
df_remap['PGs up'] = df_remap['PGs current'] + df_remap['Remap now'] + df_remap['Remap waiting']
df_util = df_util.join(df_remap[['PGs current', 'Remap now', 'Remap waiting', 'PGs up']])
df_util['util current'] = df_util['util'].round(4)
# For active remaps, assume half of PG has transferred
util_change = (0.5*df_util['Remap now'] + df_util['Remap waiting']) / df_util['PGs current']
df_util['util up'] = (df_util['util current'] * (1 + util_change)).round(4)
df_util = df_util.drop('util', axis=1)

if args.osd is not None:
    if args.osd not in df_util.index:
        print(f'Warning: specified osd "{args.osd}" is not in utilisation data.')
    else:
        print(df_util.loc[args.osd])

# Calculate deviation from mean utilisation
mean_util = df_util['util up'].mean()
print(f"# mean_util = {100*mean_util:.1f}%")
df_util['deviation'] = df_util['util up'] - mean_util
df_util['dev abs'] = df_util['deviation'].abs()
total_pgs = df_util['PGs up'].sum()

df_top = df_util.copy()

# Ignore small deviations
f_below_min_dev = df_top['dev abs']<args.min
if f_below_min_dev.any():
    print(f'Ignoring {np.sum(f_below_min_dev)} OSDs as their deviation below threshold')
    df_top = df_top[~f_below_min_dev]
    if df_top.empty:
        print("No significant deviations")
        quit()

# Discard OSDs with util < mean but weight=1.0, because no room to increase weight.
f = (df_top['weight']==1.0) & (df_top['util up']<mean_util)
if f.any():
    if np.sum(f) < 5:
        maxxed_osds = sorted(df_top.index[f].tolist())
        msg = f"Ignoring {np.sum(f)} OSDs because their util is < mean but their weight=1.0, can't increase weight: {maxxed_osds}"
    else:
        msg = f"Ignoring {np.sum(f)} OSDs because their util is < mean but their weight=1.0, can't increase weight."
    print(msg)
    # print("DEBUG: dropping:") ; print(df_top[f].sort_values('dev abs', ascending=False))
    df_top = df_top[~f]
    if df_top.empty:
        print("No significant deviations with weight<1")
        quit()

# Restrict to top-N biggest deviations
if args.limit is not None:
    print(f"Only analysing the top {args.limit} OSDs with biggest deviations")
    df_top = df_top.sort_values('dev abs', ascending=False).iloc[:args.limit]
    # print(df_top) ; quit()

# Set new weights to exactly where we want them, not a vague shift.
new_weight = df_top['weight'] * (mean_util / df_top['util up'])
df_top['weight shift'] = new_weight - df_top['weight']

# Check if a shifted weight would be > 1.
df_top['new weight'] = df_top['weight'] + df_top['weight shift']
new_weight_max = (df_top['weight'] + df_top['weight shift']).max()
new_weights = df_top[['util current', 'util up', 'weight', 'new weight', 'PGs up']].copy()
if new_weight_max <= 1:
    # Simple, just change these selected OSD
    pass
else:
    scale_factor = 1 / new_weight_max
    # print(f"WARNING: new_weight_max = {new_weight_max}")

    # # Oh dear, need to change ALL weights.
    # Because if we just downscale the top new weights, 
    # then they can be almost identical to current weights.

    # df_util_wo_top = df_util[~df_util.index.isin(df_top.index)].copy()           
    # df_util_wo_top['new weight'] = df_util_wo_top['weight'] * scale_factor
    # new_weights1 = df_top[['weight', 'new weight']]
    # new_weights2 = df_util_wo_top[['weight', 'new weight']]
    # new_weights = pd.concat([new_weights1, new_weights2])
    
    # But that may put big load on Ceph, so just cap the max.
    # But at some point, must reweight all OSDs.
    f_above_1 = new_weights['new weight'] > 1.0
    if f_above_1.any():
        print(f"Clipping {np.sum(f_above_1)} new weights to maximum 1.0")
        new_weights['new weight'] = new_weights['new weight'].clip(upper=1.0)

new_weights['new weight'] = new_weights['new weight'].round(5)
new_weights['shift'] = (new_weights['new weight'] - new_weights['weight']).round(4)
# reduce shift, because I'm worried there is still flip-flopping
print("Reducing weight shift by 25%, because I'm worried there is still flip-flopping.")
new_weights['shift'] = (new_weights['shift']*0.75).round(4) ; new_weights['new weight'] = new_weights['weight'] + new_weights['shift']
new_weights = new_weights[new_weights['shift'] != 0]
if new_weights.empty:
    print("No significant reweights needed")
    quit()
# new_weights = new_weights.sort_values('shift')
new_weights = new_weights.sort_values('util up', ascending=False)
new_weights['PGs to move'] = ((new_weights['shift'] / new_weights['weight']) * new_weights['PGs up']).round(1)
# new_weights['dev'] = (new_weights['util up'] - new_weights['util up'].mean()).abs()
cols_sorted = ['util current', 'util up', 'weight', 'new weight', 'shift', 'PGs up', 'PGs to move']
print(f"# new_weights: count={len(new_weights)}") ; print(new_weights[cols_sorted])

pgs_pushed = abs((new_weights['PGs to move'][new_weights['PGs to move']<0]).sum())
pgs_pulled = abs((new_weights['PGs to move'][new_weights['PGs to move']>0]).sum())
max_pgs_written = max(pgs_pushed, pgs_pulled)
print(f"# PGs to write = {max_pgs_written:.0f} = {100*max_pgs_written/total_pgs:.1f}%")

# Finally, write out the table and Bash script
ok = click.confirm("Generate Bash script you can run?", default=False)
if not ok:
    quit()
job_id = f'{dt.datetime.now().strftime('%Y-%m-%dT%H-%M')}'
_cwd = "/root"
df_fn = f"reweight-{job_id}.csv"
script_fn = f"reweight-{job_id}.sh"
if args.cephadm:
    # Write to local folder to mount inside cephadm
    jobs_dir = args.outdir
    if not os.path.isdir(jobs_dir):
        os.makedirs(jobs_dir)
    df_fp = os.path.join(jobs_dir, df_fn)
    script_fp = os.path.join(jobs_dir, script_fn)
else:
    df_fp = df_fn
    script_fp = script_fn
new_weights.to_csv(df_fp)
print(f"Table written to: {df_fp}")
with open(script_fp, 'w') as F:
    F.write("#!/bin/bash\n\n")
    F.write("ceph osd set norebalance\n")
    F.write("sleep 5\n")
    for i in range(len(new_weights)):
        row = new_weights.iloc[i]
        osd_id = row.name
        if isinstance(osd_id, (int, np.int64)):
            osd_id = "osd."+str(osd_id)
        weight = row['new weight']
        F.write(f"ceph osd reweight {osd_id} {weight:.5f}\n")
    #F.write("ceph osd unset norebalance\n")
    F.write('echo "To trigger rebalance run: ceph osd unset norebalance"\n')
    F.write('echo "Or instead, re-run this script to refine weights."\n')
    F.write("\n")

print(f"ceph reweight commands written to script: {script_fp}")
if args.cephadm:
    cmd = f"./cephadm shell --mount {_cwd}/{jobs_dir} -- bash /mnt/{script_fp}"
else:
    cmd = f"bash {script_fp}"
print(f"To run it: {cmd}")


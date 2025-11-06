## Motivation

This tool sits between 2 existing tools:
- Ceph's built-in tool `ceph osd reweight-by-utilization` with questionable reweight decisions
- [TheJJ/ceph-balancer modifies upmap and can violate your CRUSH rules](https://github.com/TheJJ/ceph-balancer/issues/41)

`ceph osd reweight-by-utilization` can have bias for reducing weights of high-util OSDs, and not equally consider increasing low-util OSDs. 
This tool primarily just addresses that deficiency.

Features:
- analyse Up not Current utilisation, so can run during a rebalance - query Ceph for accurate PG backfill progress
- reweight a specific pool
- exclude hostnames

## Install

Install Python dependencies: Pandas Numpy
> apt|dnf install python3-pandas python3-numpy

Then download [reweight.py](https://github.com/WIMM-IT/ceph-reweight-by-util/blob/main/reweight.py)

## Use

```
usage: reweight.py [-h] -p POOL [-m MIN] [-l LIMIT] [-d DOWNSCALE] [-o OSD] [-s] [-e EXCLUDE_HOST] [-b]

Calculate Ceph OSD reweights using deviation from mean utilisation %. Only calculate - reweight is still a
manual task for you to review. Accounts for any PGs currently being remapped (ceph pg dump pgs_brief), by
analysing the utilisation after current remap completes.

options:
  -h, --help            show this help message and exit
  -p POOL, --pool POOL  Focus on this Ceph pool
  -m MIN, --min MIN     Deviation threshold. E.g. 5 means: ignore OSDs within mean util % +-5%
  -l LIMIT, --limit LIMIT
                        Optional: limit to N OSDs with biggest deviation
  -d DOWNSCALE, --downscale DOWNSCALE
                        Downscale all weights by this amount e.g. 0.9 to reduce by 10%. To give room to handle
                        low-util OSDs with reweight already at maximum 1.
  -o OSD, --osd OSD     Optional: print detailed information for this OSD number
  -s, --cephadm         Run Ceph query commands via cephadm shell
  -e EXCLUDE_HOST, --exclude-host EXCLUDE_HOST
                        Exclude these hosts matching these regex patterns. Can be used multiple times.
  -b, --backup          Backup weights as a Bash restore script. Do nothing else
```

## Example

Reweight OSDs with util deviation 2.2% from mean

#### Create and inspect plan

> python3 reweight.py -p $pool -m 2.2 1> reweight-apply.sh

![](./Images/osds-util-annot.png)

```
# new_weights: count=15
     util current  util up   weight  new weight   shift  PGs up  PGs to move
id                                                                          
398        0.8540   0.8540  0.93924     0.91674 -0.0225      61         -1.5
20         0.8538   0.8538  0.95251     0.92991 -0.0226      60         -1.4
...
412        0.7863   0.7863  0.93964     0.97584  0.0362      55          2.1
483        0.7853   0.7853  0.95079     0.98769  0.0369      61          2.4
# PGs to write = 20 = 0.1%
```

#### Apply plan

> ./reweight-apply.sh

## Advanced

#### Downscale

It is not impossible for an OSD to have maximum reweight 1.0 but have low utilisation well below the mean.
Argument `--downscale` helps handle this situation.
Set it to 0.9 to shift all reweights down by 10%.
This creates headroom for increasing reweight on those low utilisation OSDs.

Expect significant remapping even though relatively the reweights have not changed.
This is a consequence of Ceph deriving CRUSH map with hashing.

## Motivation

Built-in tool `ceph osd reweight-by-utilization` has bias for reducing high-util OSDs. 
This seems the obvious solution until you consider where the displaced PGs go - if they go to other near-high-util OSDs then you'll need to rebalance again soon.
Instead can they be directed to the low-util OSDs by re-weighting both high-util and low-util?

This script has two other features:
- never modifies Ceph, instead it generates a Bash script you run to apply rebalance
- analyse "up" not "current" util, so can re-run during a live rebalance

Note: "up util" is a rough measure, it assumes any PGs being actively remapped are 50% transferred (not waiting). If you know how to be more accurate please let me know.

## Install

Install Python dependencies: Pandas Numpy click
> apt|dnf install python3-pandas python3-numpy python3-click

Then download and run script `reweight.py`

## Use

```
usage: reweight.py [-h] -p POOL [-m MIN] [-d OUTDIR] [-l LIMIT] [-o OSD] [-s]

Calculate Ceph OSD reweights using deviation from mean utilisation %. Only calculate - reweight is
still a manual task for you to review. Accounts for any PGs currently being remapped (ceph pg dump
pgs_brief), by analysing the utilisation after current remap completes.

options:
  -h, --help            show this help message and exit
  -p POOL, --pool POOL  Focus on this Ceph pool
  -m MIN, --min MIN     Deviation threshold. E.g. 5 means: ignore OSDs within mean util % +-5%
  -d OUTDIR, --outdir OUTDIR
                        Store reweight job script in this folder. This Python script does not actually
                        apply weights
  -l LIMIT, --limit LIMIT
                        Optional: limit to N OSDs with biggest deviation
  -o OSD, --osd OSD     Optional: print detailed information for this OSD number
  -s, --cephadm         Run Ceph query commands via cephadm shell
```

## Example

Reweight OSDs with util deviation 2.2% from mean

> python3 ./reweight.py -p $pool -m 2.2

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
Generate Bash script you can run? [y/N]:
```

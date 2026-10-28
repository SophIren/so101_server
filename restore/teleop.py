#!/usr/bin/env python3
"""Portable entry point to the supplied calibrated leader/follower teleop."""
import argparse
from pathlib import Path

from teleop_loader import load_teleop, resolve_controller


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--supported', action='store_true')
    parser.add_argument('--resume-holding', action='store_true')
    parser.add_argument('--full-range', action='store_true')
    parser.add_argument('--relative', action='store_true')
    parser.add_argument('--smooth-tracking', action='store_true')
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--speed', type=int, default=400)
    parser.add_argument('--leader-port')
    parser.add_argument('--follower-port')
    args = parser.parse_args(argv)
    if args.execute and not args.supported:
        parser.error('Support the follower and clear its workspace, then pass --supported')
    driver = load_teleop()
    leader = resolve_controller(driver.CONFIG['leader_port'], '5B79016829', args.leader_port)
    follower = resolve_controller(driver.CONFIG['follower_port'], '5B8E114736', args.follower_port)
    if Path(leader).resolve() == Path(follower).resolve():
        parser.error('Leader and follower must be connected to different serial ports')
    options = ['--leader-port', leader, '--follower-port', follower,
               '--seconds', str(args.seconds), '--speed', str(args.speed)]
    for flag in ('execute', 'resume_holding', 'full_range', 'relative', 'smooth_tracking'):
        if getattr(args, flag):
            options.append('--' + flag.replace('_', '-'))
    driver.main(options)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

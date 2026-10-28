#!/usr/bin/env python3
"""Explicitly requested test motion. Imports and --help never open devices."""
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 parser=argparse.ArgumentParser(description='SO-101 bounded jog or calibration-pose roundtrip')
 parser.add_argument('--execute',action='store_true',help='Enable motors and perform motion')
 parser.add_argument('--supported',action='store_true',help='Initial/return pose has stable mechanical support; workspace clear')
 parser.add_argument('--joint',choices=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'])
 parser.add_argument('--delta',type=int,default=24,help='Jog encoder counts, at most 90; always returns')
 args=parser.parse_args()
 if not args.execute:
  print('NO MOTION. Add --execute --supported after checking mechanical support and workspace.');return
 if not args.supported:parser.error('--supported is required: return must have mechanical support')
 if not 1<=abs(args.delta)<=90:parser.error('delta must have magnitude 1..90')
 import json,time,datetime
 from pathlib import Path
 import serial
 from verified_telemetry import read_health
 from serial_protocol import exchange
 cal=json.loads((ROOT/'so101_follower_calibration.json').read_text())
 config=json.loads((ROOT/'robot_setup/robot_config.json').read_text())
 log={'started':datetime.datetime.now().isoformat(),'stages':[]}
 path=ROOT/'logs'/('roundtrip_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
 path.parent.mkdir(exist_ok=True)
 s=serial.Serial(config['serial_port'],config['baudrate'],timeout=.15,exclusive=True)
 active=[];backup={};completed=False
 
 def save():path.write_text(json.dumps(log,indent=2)+'\n')
 def tx(i,cmd,args,n):
  def retry(event):log.setdefault('transport_retries',[]).append(event)
  return exchange(s,i,cmd,args,n,on_retry=retry)
 def read(i,a,n):return int.from_bytes(tx(i,2,[a,n],n+6),'little')
 def write(i,a,n,v):
  tx(i,3,[a]+list(v.to_bytes(n,'little')),6)
  assert read(i,a,n)==v,f'ID {i}: write failed at {a}'
 def positions():return {name:read(c['id'],56,2) for name,c in cal.items()}
 def goals(pose):
  args=[42,2]
  for name,v in pose.items():
   c=cal[name]
   assert c['range_min']<=v<=c['range_max'],f'Invalid goal {name}={v}'
   args.extend([c['id'],v&255,v>>8])
  p=[254,len(args)+2,131]+args;s.write(bytes([255,255]+p+[(~sum(p))&255]));s.flush()
 def check(goal):
  actual=positions();health={}
  for name,c in cal.items():
   i=c['id'];telemetry=read_health(i,tx);temp=telemetry['temp'];status=telemetry['status'];load=telemetry['load'];voltage=telemetry['voltage']
   if 'confirmed_temperature_outlier' in telemetry:
    log.setdefault('temperature_outliers',[]).append({'id':i,**telemetry});save();print('CONFIRMED TELEMETRY OUTLIER',i,telemetry,flush=True)
   health[name]={'temp':temp,'status':status,'load':load}
   if abs(actual[name]-goal[name])>45 or temp>=45 or status!=0 or load>=850 or not 105<=voltage<=135:
    log['stop_state']={'positions':actual,'goal':goal,'health':health};save()
    raise RuntimeError(f'{name}: tracking={actual[name]-goal[name]}, temp={temp}, status={status}, load={load}, voltage={voltage}')
  return actual
 
 def move_one(pose,name,target,label):
  start=positions()[name];assert abs(start-pose[name])<=45
  current=start;pose=dict(pose);pose[name]=start
  while current!=target:
   tick=time.monotonic();current+=max(-4,min(4,target-current));pose[name]=current;goals(pose);check(pose)
   time.sleep(max(0,.1-(time.monotonic()-tick)))
  deadline=time.monotonic()+1.5
  while time.monotonic()<deadline:
   actual=check(pose)
   if max(abs(actual[n]-pose[n]) for n in cal)<=25:break
   time.sleep(.1)
  else:raise RuntimeError(f'{name}: did not settle')
  log['stages'].append({'label':label,'joint':name,'target':target,'actual':actual});save()
  print('REACHED',label,name,target,'actual',actual[name],flush=True)
  return pose
 
 try:
  start=positions();log['start']=start
  for name,c in cal.items():
   i=c['id']
   if read(i,40,1)!=0:raise RuntimeError('Start requires torque OFF and stable mechanical support')
   if read(i,3,2)!=777 or read(i,33,1)!=0:raise RuntimeError('Unexpected model or mode')
   if read_health(i,tx)['temp']>40:raise RuntimeError('Motor too warm for preflight')
   if not c['range_min']<=start[name]<=c['range_max']:raise RuntimeError('Start outside calibration')
   off=c['homing_offset']
   if read(i,31,2)!=(abs(off)|(2048 if off<0 else 0)):raise RuntimeError('Homing offset mismatch')
   if read(i,9,2)!=c['range_min'] or read(i,11,2)!=c['range_max']:raise RuntimeError('Limits mismatch')
   backup[name]={str(a):read(i,a,n) for a,n in ((41,1),(44,2),(46,2),(48,2))}
  if args.joint and not cal[args.joint]['range_min']<=start[args.joint]+args.delta<=cal[args.joint]['range_max']:
   raise RuntimeError('Jog target outside limits; motors were not enabled')
  # Snapshot before enabling any motor. The returned pose is from THIS run.
  log['backup']=backup;save()
  for name,c in cal.items():
   i=c['id'];write(i,41,1,5);write(i,44,2,0);write(i,46,2,65)
   if name=='gripper':write(i,48,2,min(200,backup[name]['48']))
  goals(start)
  for name,c in cal.items():
   active.append(name);write(c['id'],40,1,1)
  check(start);pose=start.copy()
  if args.joint:
   c=cal[args.joint];target=start[args.joint]+args.delta
   if not c['range_min']<=target<=c['range_max']:raise RuntimeError('Jog target outside limits')
   pose=move_one(pose,args.joint,target,'jog')
   pose=move_one(pose,args.joint,start[args.joint],'return')
  else:
   for name in ('shoulder_lift','elbow_flex','wrist_flex','wrist_roll','shoulder_pan','gripper'):
    pose=move_one(pose,name,2047,'calibration')
   log['calibration_pose']=check(pose);save();time.sleep(.5)
   for name in ('gripper','shoulder_pan','wrist_roll','wrist_flex','elbow_flex','shoulder_lift'):
    pose=move_one(pose,name,start[name],'return')
  log['returned_pose']=check(start);log['success']=True;completed=True;save()
  print('COMPLETE',log['returned_pose'])
 except BaseException as error:
  log['error']=str(error)
  try:
   current=positions()
   if active:goals({n:max(c['range_min'],min(c['range_max'],current[n])) for n,c in cal.items()})
   log['stopped_pose']=current;log['holding_after_failure']=bool(active)
  except Exception as stop_error:log['stop_error']=str(stop_error)
  save();print('STOP:',error,'HOLDING:',bool(active),flush=True)
  print('Support the arm before disabling torque or power.',flush=True)
  raise
 finally:
  if completed or not active:
   for name in active:write(cal[name]['id'],40,1,0)
   for name,b in backup.items():
    for a,n in ((41,1),(44,2),(46,2),(48,2)):write(cal[name]['id'],a,n,b[str(a)])
   log['torque_after']={n:read(c['id'],40,1) for n,c in cal.items()};save()
  s.close()

if __name__=='__main__':
 main()

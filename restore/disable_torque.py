#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import serial
from so101_arm import resolve_port

def main():
 p=argparse.ArgumentParser(description='Disable all motors only with the arm physically supported')
 p.add_argument('--supported',action='store_true')
 p.add_argument('--port-name',help='Override the serial port (otherwise auto-detected; SO101_PORT is supported)')
 args=p.parse_args()
 if not args.supported:p.error('Support the arm first, then pass --supported')
 root=Path(__file__).resolve().parent;config=json.loads((root/'robot_setup/robot_config.json').read_text())
 port=resolve_port(args.port_name or config['serial_port'])
 print(f'Serial port: {port}',flush=True)
 with serial.Serial(port,config['baudrate'],timeout=.3,exclusive=True) as s:
  def tx(i,c,a,n):
   s.reset_input_buffer();p=[i,len(a)+2,c]+a;s.write(bytes([255,255]+p+[(~sum(p))&255]));s.flush();r=s.read(n)
   if len(r)!=n or r[:3]!=bytes([255,255,i]) or r[3]!=n-4 or r[4]!=0 or sum(r[2:])%256!=255:raise RuntimeError(f'Invalid response ID {i}')
   return r[5:-1]
  errors=[]
  for i in range(1,7):
   try:tx(i,3,[40,0],6)
   except Exception as e:errors.append(str(e))
  for i in range(1,7):
   try:
    if tx(i,2,[40,1],7)!=b'\x00':errors.append(f'ID {i}: torque still enabled')
   except Exception as e:errors.append(str(e))
  if errors:raise RuntimeError('; '.join(errors))
  print('Verified: all six motors torque OFF')

if __name__=='__main__':main()

const { spawn } = require('child_process');
const cmd = 'ffmpeg';
const args = [
  '-rtsp_transport', 'tcp',
  '-re',
  '-i', 'rtsp://admin:admin123@192.168.31.198:554/cam/realmonitor?channel=1&subtype=0',
  '-codec:v', 'mpeg1video',
  '-f', 'mpegts',
  '-b:v', '2000k',
  '-r', '25',
  '-q:v', '1',
  '-s', '1920x1080',
  'pipe:1'
];

const proc1 = spawn(cmd, args);
console.log('proc1 PID:', proc1.pid);

const proc2 = spawn(cmd, args);
console.log('proc2 PID:', proc2.pid);
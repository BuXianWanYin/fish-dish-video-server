const express = require('express');
const WebSocket = require('ws');
const ffmpeg = require('fluent-ffmpeg');
const mysql = require('mysql2/promise');
const path = require('path');
const cors = require('cors');

const app = express();         // 先初始化 app
app.use(cors());               // 再 use cors
ffmpeg.setFfmpegPath('D:/BaiduNetdiskDownload/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe'); // 你的ffmpeg路径

const httpPort = 9999;

// 数据库连接池
const db = mysql.createPool({
    host: '192.168.31.37',
    user: 'root',
    password: '123456',
    database: 'fish-dish-server',
    port: 3306, 
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
  });

// 静态托管前端
app.use(express.static(path.join(__dirname, 'dist')));

// 提供前端获取 ws 地址的接口
app.get('/api/stream/:deviceId', async (req, res) => {
  const deviceId = req.params.deviceId;
  const [rows] = await db.query('SELECT * FROM agriculture_camera WHERE device_id=?', [deviceId]);
  if (!rows.length) return res.status(404).json({ error: '设备不存在' });
  res.json({ wsUrl: `/ws/stream/${deviceId}` });
});

const server = app.listen(httpPort, () => {
  console.log(`HTTP server running at http://localhost:${httpPort}`);
});

// WebSocket 服务
const wsServer = new WebSocket.Server({ server });

wsServer.on('connection', async (ws, req) => {
  // 解析 deviceId
  const url = req.url; // 例如 /ws/stream/1
  const match = url.match(/\/ws\/stream\/(\d+)/);
  if (!match) {
    ws.close();
    return;
  }
  const deviceId = match[1];

  // 查数据库
  const [rows] = await db.query('SELECT * FROM agriculture_camera WHERE device_id=?', [deviceId]);
  if (!rows.length) {
    ws.close();
    return;
  }
  const device = rows[0];
  const rtspUrl = `rtsp://${device.username}:${device.password}@${device.ip}:${device.port}/cam/realmonitor?channel=${device.channel}&subtype=${device.subtype}`;
  console.log('拉流地址:', rtspUrl);

  // ffmpeg 拉流推送
  const ffmpegCommand = ffmpeg(rtspUrl)
    .inputOptions([
      '-rtsp_transport tcp',
      '-re'
    ])
    .outputOptions([
      '-codec:v mpeg1video',
      '-f mpegts',
      '-b:v 2000k',
      '-r 25',
      '-q:v 1',
      '-s 1920x1080'
    ])
    .on('start', (commandLine) => {
      console.log('FFmpeg started:', commandLine);
    })
    .on('error', (err) => {
      console.error('FFmpeg Error:', err.message);
      ws.close();
    })
    .on('end', () => {
      console.log('FFmpeg ended');
      ws.close();
    });

  const stream = ffmpegCommand.pipe();

  stream.on('data', (data) => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(data);
    }
  });

  ws.on('close', () => {
    console.log('WebSocket closed');
    ffmpegCommand.kill('SIGKILL');
  });
});

process.on('SIGINT', () => {
  console.log('正在关闭服务器...');
  wsServer.close(() => {
    process.exit(0);
  });
}); 
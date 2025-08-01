const express = require('express');
const WebSocket = require('ws');
const { spawn } = require('child_process');
const mysql = require('mysql2/promise');
const path = require('path');
const cors = require('cors');

const app = express();         // 先初始化 app
app.use(cors());               // 再 use cors
// ffmpeg.setFfmpegPath('D:/BaiduNetdiskDownload/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe'); // 你的ffmpeg路径

// 服务器配置
const httpHost = '192.168.31.168';
const httpPort = 9999;

// 数据库配置
const dbHost = '192.168.31.37';
const dbUser = 'root';
const dbPassword = '123456';
const dbName = 'fish-dish-server';
const dbPort = 3306;

// 数据库连接池
const db = mysql.createPool({
    host: dbHost,
    user: dbUser,
    password: dbPassword,
    database: dbName,
    port: dbPort, 
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
  console.log(`HTTP 服务器已启动，地址: http://${httpHost}:${httpPort}`);
});

// WebSocket 服务
const wsServer = new WebSocket.Server({ server });
let wsCount = 0;
// 多播流管理：deviceId -> { ffmpegProc, clients(Set), broadcast }
const streams = new Map();

// === 新增: 定时 ping/pong 机制 ===
const interval = setInterval(() => {
  wsServer.clients.forEach((ws) => {
    if (ws.isAlive === false) {
      ws.terminate();
      return;
    }
    ws.isAlive = false;
    ws.ping();
  });
}, 30000); // 30秒检测一次
// === 新增结束 ===

wsServer.on('connection', async (ws, req) => {
  wsCount++;
  // === 新增: 每个 ws 连接加 isAlive 标志和 pong 监听 ===
  ws.isAlive = true;
  ws.on('pong', () => {
    ws.isAlive = true;
  });
  // === 新增结束 ===
  console.log('当前活跃 ws 数量:', wsCount);
  console.log('新的 WebSocket 连接:', req.url, '时间:', new Date());
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

  // 多播逻辑
  let streamObj = streams.get(deviceId);
  if (!streamObj) {
    // 第一个客户端，新建 ffmpeg 进程和客户端集合
    const ffmpegArgs = [
      '-rtsp_transport', 'tcp',
      '-re',
      '-i', rtspUrl,
      '-codec:v', 'mpeg1video',
      '-f', 'mpegts',
      '-b:v', '2000k',
      '-r', '25',
      '-q:v', '1',
      '-s', '1920x1080',
      'pipe:1'
    ];
    const ffmpegProc = spawn('ffmpeg', ffmpegArgs);
    console.log('FFmpeg 子进程 PID:', ffmpegProc.pid, 'deviceId:', deviceId);
    const clients = new Set();
    // 广播函数
    const broadcast = (data) => {
      for (const client of clients) {
        if (client.readyState === WebSocket.OPEN) {
          client.send(data);
        }
      }
    };
    ffmpegProc.stdout.on('data', broadcast);
    ffmpegProc.stderr.on('data', (data) => {
      // 可选：打印 ffmpeg 日志
      // console.error(`FFmpeg stderr: ${data}`);
    });
    ffmpegProc.on('close', (code, signal) => {
      console.log(`FFmpeg 进程关闭，PID: ${ffmpegProc.pid}, code: ${code}, signal: ${signal}`);
      // 通知所有客户端关闭
      for (const client of clients) {
        try { client.close(); } catch (e) {}
      }
      streams.delete(deviceId);
    });
    streamObj = { ffmpegProc, clients, broadcast };
    streams.set(deviceId, streamObj);
  }
  // 加入客户端集合
  streamObj.clients.add(ws);

  ws.on('close', () => {
    wsCount--; 
    streamObj.clients.delete(ws);
    console.log('WebSocket 连接已关闭，deviceId:', deviceId, '剩余客户端:', streamObj.clients.size);
    if (streamObj.clients.size === 0) {
      // 没有客户端了，关闭 ffmpeg
      streamObj.ffmpegProc.kill('SIGKILL');
      streams.delete(deviceId);
    }
  });
});

process.on('SIGINT', () => {
  console.log('正在关闭服务器...');
  // === 新增: 清理定时器 ===
  clearInterval(interval);
  // === 新增结束 ===
  wsServer.close(() => {
    process.exit(0);
  });
}); 
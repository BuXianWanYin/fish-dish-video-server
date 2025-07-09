# Fish Dish Video Server

一个基于 Node.js 的 RTSP 视频流服务器，支持将 RTSP 摄像头流转换为 WebSocket 流，供前端实时播放。

## 功能特性

- 🎥 **RTSP 流转换**: 将 RTSP 摄像头流转换为 WebSocket 流
- 🌐 **WebSocket 实时传输**: 支持多客户端同时观看
- 🎮 **前端播放器**: 基于 JSMpeg 的 Web 播放器
- 🗄️ **数据库支持**: MySQL 数据库管理摄像头设备信息
- 🔧 **FFmpeg 集成**: 使用 FFmpeg 进行视频转码
- 🚀 **CORS 支持**: 跨域资源共享支持

## 技术栈

- **后端**: Node.js + Express
- **WebSocket**: ws 库
- **视频处理**: FFmpeg + fluent-ffmpeg
- **数据库**: MySQL
- **前端播放器**: JSMpeg
- **跨域**: CORS

## 项目结构

```
fish-dish-video-server/
├── server.js          # 主服务器文件
├── demo.html          # 前端播放器示例
├── package.json       # 项目依赖
├── start.bat          # Windows 启动脚本
├── jsmpeg/            # JSMpeg 播放器库
├── node_modules/      # Node.js 依赖包
└── README.md          # 项目说明文档
```

## 安装要求

### 系统要求
- Node.js (推荐 v16 或更高版本)
- MySQL 数据库
- FFmpeg

### FFmpeg 安装
1. 下载 FFmpeg: https://ffmpeg.org/download.html
2. 解压到本地目录
3. 在 `server.js` 中更新 FFmpeg 路径：
   ```javascript
   ffmpeg.setFfmpegPath('你的ffmpeg路径/bin/ffmpeg.exe');
   ```

## 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/BuXianWanYin/fish-dish-video-server.git
   cd fish-dish-video-server
   ```

2. **安装依赖**
   ```bash
   npm install
   ```

3. **配置数据库**
   - 创建 MySQL 数据库 `fish-dish-server`
   - 创建表 `agriculture_camera`，包含以下字段：
     - `device_id` (设备ID)
     - `ip` (摄像头IP地址)
     - `port` (端口)
     - `username` (用户名)
     - `password` (密码)
     - `channel` (通道号)
     - `subtype` (子类型)

4. **修改数据库配置**
   在 `server.js` 中更新数据库连接信息：
   ```javascript
   const db = mysql.createPool({
       host: '你的数据库IP',
       user: '用户名',
       password: '密码',
       database: 'fish-dish-server',
       port: 3306
   });
   ```

## 使用方法

### 启动服务器
```bash
# 方法1: 直接启动
node server.js

# 方法2: 使用批处理文件 (Windows)
start.bat
```

### 访问前端
1. 打开浏览器访问: `http://localhost:9999/demo.html`
2. 修改 `demo.html` 中的设备ID (默认是50)
3. 确保数据库中有对应的设备信息

### API 接口

#### 获取设备流地址
```
GET /api/stream/:deviceId
```

**响应示例:**
```json
{
  "wsUrl": "/ws/stream/50"
}
```

#### WebSocket 流地址
```
ws://localhost:9999/ws/stream/:deviceId
```

## 配置说明

### 视频参数
在 `server.js` 中可以调整以下视频参数：
- 分辨率: `-s 1920x1080`
- 码率: `-b:v 2000k`
- 帧率: `-r 25`
- 编码: `mpeg1video`

### 端口配置
默认端口为 9999，可在 `server.js` 中修改：
```javascript
const httpPort = 9999;
```

## 故障排除

### 常见问题

1. **WebSocket 连接失败**
   - 检查服务器是否启动
   - 确认端口 9999 未被占用
   - 检查防火墙设置

2. **FFmpeg 错误**
   - 确认 FFmpeg 路径正确
   - 检查 RTSP 地址是否可访问
   - 验证摄像头凭据

3. **数据库连接失败**
   - 检查数据库服务是否运行
   - 确认连接参数正确
   - 验证数据库和表是否存在

### 日志查看
服务器启动后会在控制台输出详细日志，包括：
- HTTP 服务器状态
- WebSocket 连接信息
- FFmpeg 转码状态
- 错误信息

## 开发说明

### 添加新设备
1. 在数据库 `agriculture_camera` 表中添加设备信息
2. 前端修改设备ID即可观看

### 自定义前端
可以基于 `demo.html` 开发自定义的前端界面，主要需要：
- 引入 `jsmpeg.min.js`
- 创建 canvas 元素
- 配置 JSMpeg.Player

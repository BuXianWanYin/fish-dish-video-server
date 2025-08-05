# RTSP视频流推流服务器

一个基于Python的高性能RTSP视频流获取和WebSocket推流服务器，支持多客户端实时观看。

## 🚀 特性

- **高性能异步处理**: 基于FastAPI和asyncio的异步架构
- **多客户端支持**: 支持多个客户端同时观看同一视频流
- **智能资源管理**: 自动管理连接和线程资源，无客户端时自动停止处理
- **实时监控**: 提供FPS、连接状态等实时监控信息
- **无FFmpeg依赖**: 使用OpenCV直接处理RTSP流，简化部署
- **WebSocket实时推送**: 低延迟的视频帧推送

## 🛠️ 技术栈

| 组件 | 技术选择 |
|------|----------|
| 后端框架 | FastAPI + asyncio |
| 视频处理 | OpenCV |
| 数据库 | MySQL (aiomysql) |
| 前端 | HTML5 + JavaScript + Canvas |
| 通信协议 | WebSocket |

## 📦 安装

### 环境要求
- Python 3.8+
- MySQL 5.7+

### 安装依赖
```bash
pip install -r requirements.txt
```

## ⚙️ 配置

编辑 `main.py` 文件中的配置参数：

```python
# 服务器配置
HTTP_HOST = '192.168.31.168'  # 服务器IP地址
HTTP_PORT = 9999               # HTTP服务端口

# 数据库配置
DB_HOST = '192.168.31.37'     # 数据库主机
DB_USER = 'root'               # 数据库用户名
DB_PASSWORD = '123456'         # 数据库密码
DB_NAME = 'fish-dish-server'   # 数据库名称
DB_PORT = 3306                 # 数据库端口
```

## 🚀 运行

```bash
python main.py
```

启动后访问：
- **HTTP API**: `http://192.168.31.168:9999`
- **WebSocket**: `ws://192.168.31.168:9998`
- **测试页面**: `http://192.168.31.168:9999/index.html`

## 📡 API接口

### 获取设备流信息
```http
GET /api/stream/{device_id}
```

**响应示例**:
```json
{
  "wsUrl": "/ws/stream/{device_id}"
}
```

## 🔌 WebSocket协议

### 发送消息
```json
{
  "type": "ping"
}
```

### 接收消息
```json
{
  "type": "frame",
  "deviceId": "1",
  "data": "base64编码的JPEG图像",
  "timestamp": 1234567890.123
}
```

## 🗄️ 数据库设计

### agriculture_camera 表结构

| 字段名 | 类型 | 说明 |
|--------|------|------|
| device_id | VARCHAR | 设备唯一标识 |
| ip | VARCHAR | 摄像头IP地址 |
| username | VARCHAR | 登录用户名 |
| password | VARCHAR | 登录密码 |
| port | INT | RTSP端口 |
| channel | INT | 视频通道号 |
| subtype | VARCHAR | 设备子类型 |

## 🖥️ 前端使用

1. 打开浏览器访问 `index.html`
2. 输入要观看的设备ID
3. 点击"连接视频流"按钮
4. 等待连接建立，开始观看实时视频

## ⚡ 性能优化

- **视频质量**: 1280x720分辨率，80% JPEG质量
- **帧率控制**: 25FPS稳定输出
- **并发处理**: 线程池处理RTSP流
- **内存管理**: 自动清理断开的连接和资源

## 🔧 故障排除

### 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 无法连接RTSP流 | 摄像头配置错误 | 检查IP、用户名、密码配置 |
| WebSocket连接失败 | 防火墙阻止 | 确保9998端口开放 |
| 数据库连接失败 | 网络或配置问题 | 检查数据库配置和网络连接 |
| 视频显示异常 | 浏览器兼容性 | 确保浏览器支持WebSocket和Canvas |

### 日志查看
服务器运行时会输出详细的连接和错误日志，便于问题诊断。

**注意**: 请确保在使用前正确配置数据库和摄像头信息。 
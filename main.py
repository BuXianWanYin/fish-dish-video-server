import asyncio
import json
import cv2
import base64
import aiomysql
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from websockets.server import serve, WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed
import threading
import time
from typing import Dict, Set, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入配置
from config import HTTP_HOST, HTTP_PORT, DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

# 全局变量
app = FastAPI()
streams: Dict[str, Dict] = {}  # deviceId -> {clients, thread, running}
ws_count = 0

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 导入API路由
from api_routes import router as api_router

# 注册API路由
app.include_router(api_router)

# 静态文件服务 - 只挂载到特定路径，避免拦截API和WebSocket
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

@app.get("/")
async def read_index():
    return FileResponse("index.html")

@app.get("/index.html")
async def read_index_html():
    return FileResponse("index.html")

# 数据库连接池
async def get_db_pool():
    return await aiomysql.create_pool(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
        port=DB_PORT,
        autocommit=True,
        maxsize=10
    )

# RTSP流处理类
class RTSPStreamHandler:
    def __init__(self, device_id: str, rtsp_url: str):
        self.device_id = device_id
        self.rtsp_url = rtsp_url
        self.clients: Set[WebSocketServerProtocol] = set()
        self.running = False
        self.thread = None
        # 存储主事件循环的引用
        try:
            self.main_loop = asyncio.get_event_loop()
        except RuntimeError:
            self.main_loop = None
        
    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._stream_worker)
            self.thread.daemon = True
            self.thread.start()
            logger.info(f"开始RTSP流处理，设备ID: {self.device_id}")
    
    def stop(self):
        self.running = False
        if self.thread and self.thread != threading.current_thread():
            self.thread.join(timeout=5)
        logger.info(f"停止RTSP流处理，设备ID: {self.device_id}")
    
    def add_client(self, client: WebSocketServerProtocol):
        self.clients.add(client)
        if not self.running:
            self.start()
        logger.info(f"添加客户端到设备 {self.device_id}，当前客户端数: {len(self.clients)}")
    
    def remove_client(self, client: WebSocketServerProtocol):
        self.clients.discard(client)
        logger.info(f"移除客户端从设备 {self.device_id}，当前客户端数: {len(self.clients)}")
        
        # 如果没有客户端了，停止流处理
        if len(self.clients) == 0:
            logger.info(f"设备 {self.device_id} 没有更多客户端，准备停止流处理")
            self.stop()
            if self.device_id in streams:
                del streams[self.device_id]
                logger.info(f"已从全局流字典中移除设备 {self.device_id}")
    
    def _stream_worker(self):
        """RTSP流处理工作线程"""
        cap = None
        try:
            # 设置OpenCV参数
            cap = cv2.VideoCapture(self.rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 设置RTSP传输协议为TCP
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            # 设置更多OpenCV参数以提高连接稳定性
            # 注意：某些OpenCV版本可能不支持CAP_PROP_TIMEOUT
            try:
                cap.set(cv2.CAP_PROP_TIMEOUT, 5000)  # 5秒超时
            except AttributeError:
                logger.warning("当前OpenCV版本不支持CAP_PROP_TIMEOUT，跳过此设置")
            except Exception as e:
                logger.warning(f"设置CAP_PROP_TIMEOUT失败: {e}")
            
            cap.set(cv2.CAP_PROP_FPS, 25)  # 设置帧率
            
            if not cap.isOpened():
                logger.error(f"无法打开RTSP流: {self.rtsp_url}")
                return
            
            logger.info(f"成功连接到RTSP流: {self.rtsp_url}")
            
            # 测试读取第一帧
            ret, test_frame = cap.read()
            if not ret:
                logger.error(f"无法读取RTSP流的第一帧: {self.rtsp_url}")
                return
            else:
                logger.info(f"成功读取第一帧，帧大小: {test_frame.shape}")
            
            frame_count = 0
            last_time = time.time()
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"读取RTSP帧失败，设备ID: {self.device_id}")
                    time.sleep(1)
                    continue
                
                # 计算FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    current_time = time.time()
                    fps = 30 / (current_time - last_time)
                    last_time = current_time
                    logger.info(f"设备 {self.device_id} FPS: {fps:.2f}, 帧大小: {frame.shape}")
                
                # 调整帧大小以优化传输
                frame = cv2.resize(frame, (1280, 720))
                
                # 编码为JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                jpeg_data = buffer.tobytes()
                
                # 转换为base64
                base64_data = base64.b64encode(jpeg_data).decode('utf-8')
                
                # 调试信息：每100帧记录一次
                if frame_count % 100 == 0:
                    logger.info(f"设备 {self.device_id} 已处理 {frame_count} 帧，JPEG大小: {len(jpeg_data)} 字节")
                
                # 创建消息
                message = {
                    'type': 'frame',
                    'deviceId': self.device_id,
                    'data': base64_data,
                    'timestamp': time.time()
                }
                
                # 发送给所有客户端
                message_str = json.dumps(message)
                clients_to_remove = []
                sent_count = 0
                
                for client in self.clients:
                    try:
                        if client.open:
                            # 使用存储的主事件循环发送消息
                            if self.main_loop and self.main_loop.is_running():
                                asyncio.run_coroutine_threadsafe(
                                    client.send(message_str), 
                                    self.main_loop
                                )
                                sent_count += 1
                            else:
                                logger.warning(f"主事件循环不可用，跳过发送")
                                clients_to_remove.append(client)
                        else:
                            clients_to_remove.append(client)
                    except Exception as e:
                        logger.error(f"发送帧到客户端失败: {e}")
                        clients_to_remove.append(client)
                
                # 调试信息：每100帧记录一次发送状态
                if frame_count % 100 == 0 and sent_count > 0:
                    logger.info(f"设备 {self.device_id} 成功发送帧到 {sent_count} 个客户端")
                
                # 移除断开的客户端
                for client in clients_to_remove:
                    self.remove_client(client)
                
                # 控制帧率
                time.sleep(0.04)  # 约25 FPS
                
        except Exception as e:
            logger.error(f"RTSP流处理错误，设备ID {self.device_id}: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
        finally:
            if cap:
                cap.release()
            logger.info(f"RTSP流处理结束，设备ID: {self.device_id}")

# WebSocket处理
async def websocket_handler(websocket: WebSocketServerProtocol, path: str):
    global ws_count
    
    ws_count += 1
    logger.info(f"新的WebSocket连接: {path}, 当前连接数: {ws_count}")
    
    try:
        # 解析设备ID
        if not path.startswith('/ws/stream/'):
            await websocket.close(1008, "无效的路径")
            return
        
        device_id = path.split('/')[-1]
        logger.info(f"请求设备ID: {device_id}")
        
        # 查询数据库获取设备信息
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    'SELECT * FROM agriculture_camera WHERE device_id=%s', 
                    (device_id,)
                )
                rows = await cursor.fetchall()
        
        if not rows:
            await websocket.close(1008, "设备不存在")
            return
        
        device = rows[0]
        # 根据日志分析，数据库字段顺序为: device_id, ip, username, password, port, channel, subtype
        # 但是port字段实际存储的是IP地址，需要重新构建RTSP地址
        username = device[2]
        password = device[3]
        ip = device[4]  # 实际IP地址在port字段
        port = device[5]  # 默认RTSP端口
        channel = device[6]
        subtype = device[7]
        
        # 构建正确的RTSP地址
        rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}"
        logger.info(f"RTSP地址: {rtsp_url}")
        
        # 添加调试信息
        logger.info(f"设备信息: device_id={device[0]}, ip={device[1]}, username={device[2]}, password={device[3]}, port字段={device[4]}, channel={device[5]}, subtype={device[6]}")
        logger.info(f"解析后: username={username}, password={password}, ip={ip}, port={port}, channel={channel}, subtype={subtype}")
        
        # 获取或创建流处理器
        if device_id not in streams:
            streams[device_id] = RTSPStreamHandler(device_id, rtsp_url)
        
        stream_handler = streams[device_id]
        stream_handler.add_client(websocket)
        
        # 发送连接确认消息
        await websocket.send(json.dumps({
            'type': 'connection',
            'deviceId': device_id,
            'message': '连接成功'
        }))
        
        # 保持连接
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get('type') == 'ping':
                    await websocket.send(json.dumps({'type': 'pong'}))
            except json.JSONDecodeError:
                logger.warning(f"收到无效JSON消息: {message}")
                
    except ConnectionClosed:
        logger.info("WebSocket连接已关闭")
    except Exception as e:
        logger.error(f"WebSocket处理错误: {e}")
    finally:
        ws_count -= 1
        logger.info(f"WebSocket连接关闭，剩余连接数: {ws_count}")
        
        # 从流处理器中移除客户端
        if 'device_id' in locals() and device_id in streams:
            streams[device_id].remove_client(websocket)

# 启动WebSocket服务器
async def start_websocket_server():
    # 配置WebSocket服务器，允许所有来源
    async def process_request(path, headers):
        # 允许所有来源的WebSocket连接
        return None
    
    async with serve(websocket_handler, HTTP_HOST, 9998, process_request=process_request):
        logger.info(f"WebSocket服务器已启动: ws://{HTTP_HOST}:9998")
        await asyncio.Future()  # 保持运行

# 主函数
async def main():
    # 启动WebSocket服务器
    websocket_task = asyncio.create_task(start_websocket_server())
    
    # 启动HTTP服务器
    config = uvicorn.Config(
        app, 
        host=HTTP_HOST, 
        port=HTTP_PORT,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    # 并发运行两个服务器
    await asyncio.gather(
        server.serve(),
        websocket_task
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("正在关闭服务器...")
        # 清理所有流
        for device_id, stream_handler in streams.items():
            stream_handler.stop()
        logger.info("服务器已关闭")

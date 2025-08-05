from fastapi import APIRouter, HTTPException
import aiomysql

# 导入配置
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

# 创建路由器
router = APIRouter(prefix="/api", tags=["API"])

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

@router.get("/stream/{device_id}")
async def get_stream_info(device_id: str):
    """获取设备流信息"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                'SELECT * FROM agriculture_camera WHERE device_id=%s', 
                (device_id,)
            )
            rows = await cursor.fetchall()
            
    if not rows:
        raise HTTPException(status_code=404, detail="设备不存在")
    
    return {"wsUrl": f"/ws/stream/{device_id}"} 
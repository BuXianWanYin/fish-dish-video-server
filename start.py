#!/usr/bin/env python3
"""
RTSP视频流推流服务器启动脚本
"""

import sys
import subprocess
import importlib.util
import asyncio
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """检查必要的依赖包"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'websockets',
        'opencv-python',
        'aiomysql'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        # 处理包名中的连字符
        import_name = package.replace('-', '_')
        if import_name == 'opencv_python':
            import_name = 'cv2'
        
        try:
            importlib.util.find_spec(import_name)
            logger.info(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} 未安装")
    
    if missing_packages:
        logger.error(f"缺少以下依赖包: {missing_packages}")
        logger.info("请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_opencv():
    """检查OpenCV是否正常工作"""
    try:
        import cv2
        logger.info(f"OpenCV版本: {cv2.__version__}")
        return True
    except Exception as e:
        logger.error(f"OpenCV检查失败: {e}")
        return False

async def test_database_connection():
    """测试数据库连接"""
    try:
        from main import get_db_pool
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT 1")
                result = await cursor.fetchone()
                if result:
                    logger.info("✓ 数据库连接成功")
                    return True
    except Exception as e:
        logger.error(f"✗ 数据库连接失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("=== RTSP视频流推流服务器启动检查 ===")
    
    # 检查依赖
    logger.info("\n1. 检查Python依赖包...")
    if not check_dependencies():
        sys.exit(1)
    
    # 检查OpenCV
    logger.info("\n2. 检查OpenCV...")
    if not check_opencv():
        sys.exit(1)
    
    # 测试数据库连接
    logger.info("\n3. 测试数据库连接...")
    if not asyncio.run(test_database_connection()):
        logger.warning("数据库连接失败，但服务器仍可启动")
    
    # 启动服务器
    logger.info("\n4. 启动服务器...")
    try:
        import main
        asyncio.run(main.main())
    except KeyboardInterrupt:
        logger.info("\n服务器已停止")
    except Exception as e:
        logger.error(f"服务器启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
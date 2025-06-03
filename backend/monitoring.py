import time
import logging
from functools import wraps
from typing import Dict, Any
import psutil
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackendMonitor:
    """백엔드 성능 모니터링 클래스"""
    
    @staticmethod
    def api_timer(endpoint_name: str):
        """API 응답 시간 측정 데코레이터"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = (time.time() - start_time) * 1000  # ms 단위
                    
                    # 성능 로그
                    if duration > 3000:  # 3초 이상
                        logger.warning(f"🐌 Slow API: {endpoint_name} took {duration:.2f}ms")
                    elif duration > 1000:  # 1초 이상
                        logger.info(f"⚠️ {endpoint_name}: {duration:.2f}ms")
                    else:
                        logger.info(f"⚡ {endpoint_name}: {duration:.2f}ms")
                    
                    return result
                    
                except Exception as e:
                    duration = (time.time() - start_time) * 1000
                    logger.error(f"🔥 Error in {endpoint_name} after {duration:.2f}ms: {str(e)}")
                    raise
                    
            return wrapper
        return decorator
    
    @staticmethod
    def log_system_resources():
        """시스템 리소스 사용량 로그"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        if cpu_percent > 80:
            logger.warning(f"🔥 High CPU usage: {cpu_percent}%")
        if memory.percent > 80:
            logger.warning(f"🔥 High Memory usage: {memory.percent}%")
            
        logger.info(f"💻 CPU: {cpu_percent}%, Memory: {memory.percent}%")
    
    @staticmethod
    def cache_hit_rate(cache_hits: int, total_requests: int) -> float:
        """캐시 히트율 계산"""
        if total_requests == 0:
            return 0.0
        
        hit_rate = (cache_hits / total_requests) * 100
        
        if hit_rate < 50:
            logger.warning(f"📉 Low cache hit rate: {hit_rate:.1f}%")
        else:
            logger.info(f"📈 Cache hit rate: {hit_rate:.1f}%")
            
        return hit_rate

# 단순한 인메모리 캐시 (프로덕션에서는 Redis 사용 권장)
class SimpleCache:
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.access_count: Dict[str, int] = {}
        
    def get(self, key: str) -> Any:
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        if len(self.cache) >= self.max_size:
            # LRU 방식으로 가장 적게 사용된 항목 제거
            least_used = min(self.access_count.items(), key=lambda x: x[1])
            del self.cache[least_used[0]]
            del self.access_count[least_used[0]]
        
        self.cache[key] = value
        self.access_count[key] = 1
    
    def clear(self) -> None:
        self.cache.clear()
        self.access_count.clear()
    
    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": len([k for k in self.access_count if self.access_count[k] > 1])
        }

# 전역 캐시 인스턴스
app_cache = SimpleCache(max_size=500) 
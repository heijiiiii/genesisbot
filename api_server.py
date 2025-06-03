from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import logging
import gc
import ast
import numpy as np
from pathlib import Path
import re

# gen_chatbot.py에서 필요한 것들 임포트
from gen_chatbot import (
    compiled_graph, 
    config,
    llm,
    AgentState,
    cohere_embeddings,
    client
)
from langchain_core.messages import HumanMessage, AIMessage

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Genesis G80 AI Assistant API",
    description="제네시스 G80 매뉴얼 전용 AI 어시스턴트",
    version="1.0.0"
)

# CORS 설정 (프론트엔드 연결 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용 - 프로덕션에서는 실제 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청/응답 모델 (원본 app.py 기반으로 개선)
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []
    debug_mode: Optional[bool] = False

class ChatResponse(BaseModel):
    answer: str
    context: str = ""
    images: Optional[List[Dict[str, Any]]] = None
    debug_info: Optional[Dict[str, Any]] = None
    success: bool = True
    error: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    page_filter: Optional[str] = None
    limit: Optional[int] = 5

# 메모리에 대화 세션 저장
chat_sessions = {}

# 이미지 관련성 분석 함수 (metadata 완전 제거)
def analyze_image_relevance(image_url, query_text):
    """이미지와 질문 텍스트의 관련성을 분석합니다. (metadata 사용 안함)"""
    try:
        query_embedding = cohere_embeddings.embed_query(query_text)
        
        img_embedding = None
        try:
            # image_url 컬럼으로만 검색 (metadata 사용 안함)
            resp = client.table("image_embeddings").select("embedding").eq("image_url", image_url).execute()
            if resp and resp.data and len(resp.data) > 0:
                if 'embedding' in resp.data[0]:
                    embedding_str = resp.data[0]['embedding']
                    if isinstance(embedding_str, str):
                        try:
                            embedding_str = embedding_str.replace(" ", "")
                            img_embedding = ast.literal_eval(embedding_str)
                        except:
                            logger.warning(f"임베딩 문자열 파싱 실패: {embedding_str[:50]}...")
                            img_embedding = None
                    else:
                        img_embedding = embedding_str
        except Exception as e:
            logger.error(f"이미지 임베딩 검색 오류: {str(e)}")
        
        # 임베딩 유사도 계산
        embedding_similarity = 0.5
        if img_embedding:
            try:
                norm_q = np.linalg.norm(query_embedding)
                norm_img = np.linalg.norm(img_embedding)
                if norm_q > 0 and norm_img > 0:
                    embedding_similarity = np.dot(query_embedding, img_embedding) / (norm_q * norm_img)
                    embedding_similarity = (embedding_similarity + 1) / 2  # 0~1 범위로 정규화
                    embedding_similarity = float(embedding_similarity)
            except Exception as e:
                logger.error(f"유사도 계산 오류: {str(e)}")
                embedding_similarity = 0.5
        
        # URL에서 위치 정보 추출 (metadata 대신 URL 패턴 사용)
        vertical_position = 0.5
        url_lower = image_url.lower()
        if "top" in url_lower or "upper" in url_lower:
            vertical_position = 0.2
        elif "bottom" in url_lower or "lower" in url_lower or "bot" in url_lower:
            vertical_position = 0.8
        elif "mid" in url_lower or "middle" in url_lower:
            vertical_position = 0.5
        
        result = {
            "vertical_position": float(vertical_position),
            "relevance_score": float(embedding_similarity),
            "embedding_similarity": float(embedding_similarity)
        }
        
        # 메모리 정리
        del query_embedding
        del img_embedding
        gc.collect()
        
        return result
        
    except Exception as e:
        logger.error(f"이미지 분석 오류: {str(e)}")
        return {
            "vertical_position": 0.5,
            "relevance_score": 0.5,
            "embedding_similarity": 0.5
        }

def get_page_images(page, query_text, limit=3):
    """페이지별 이미지를 검색하고 관련성을 분석합니다."""
    try:
        # URL 패턴으로 검색 (p{page}_ 패턴)
        resp = client.table("image_embeddings").select("*").ilike("image_url", f"%p{page}_%").execute()
        
        if not resp or not resp.data or len(resp.data) == 0:
            logger.warning(f"페이지 {page}에 대한 이미지를 찾을 수 없습니다.")
            return []
        
        page_images = []
        for item in resp.data:
            if 'image_url' in item and item['image_url']:
                img_url = item['image_url']
                
                # URL에서 페이지 정보 추출 (p숫자_ 패턴)
                page_match = re.search(r'p(\d+)_', img_url)
                if page_match:
                    img_page = page_match.group(1)
                else:
                    img_page = str(page)  # 패턴을 찾지 못하면 기본값 사용
                
                # 이미지 관련성 분석
                img_analysis = analyze_image_relevance(img_url, query_text)
                
                # 이미지 정보 저장
                image_info = {
                    "url": img_url,
                    "page": img_page,
                    "relevance_score": float(img_analysis["relevance_score"]),
                    "text_relevance": float(img_analysis["embedding_similarity"]),
                    "score": float(0.7 * img_analysis["relevance_score"] + 0.3)
                }
                
                page_images.append(image_info)
        
        # 관련성 점수 기준 내림차순 정렬
        page_images.sort(key=lambda x: x["relevance_score"], reverse=True)
        return page_images[:limit]
        
    except Exception as e:
        logger.error(f"페이지 이미지 검색 오류: {str(e)}")
        return []

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """개선된 채팅 API 엔드포인트"""
    try:
        logger.info(f"새로운 채팅 요청: {request.message[:50]}...")
        
        # 세션 ID 생성
        session_id = "default_session"
        
        # 기존 세션이 없으면 초기화
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                "messages": [],
                "context": "",
                "conversation_history": []
            }
        
        # 현재 세션 가져오기
        session = chat_sessions[session_id]
        
        # 새 메시지 추가
        session["messages"].append(HumanMessage(content=request.message))
        
        # LangGraph로 검색 및 응답 생성
        state = {
            "messages": session["messages"],
            "context": "",
            "conversation_history": session["conversation_history"],
            "debug_info": {}
        }
        
        # AI 응답 생성
        result = compiled_graph.invoke(state, config=config)
        
        # 마지막 AI 메시지 추출
        last_ai_message = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break
        
        if not last_ai_message:
            return ChatResponse(
                answer="죄송합니다. 응답을 생성할 수 없습니다.",
                success=False,
                error="No AI response found"
            )
        
        answer = last_ai_message.content
        context = result.get("context", "")
        
        # 디버그 정보에서 페이지 정보 추출하여 이미지 검색
        images = []
        debug_info = result.get("debug_info", {})
        
        # 검색 결과에서 페이지 정보 추출
        if "results" in debug_info:
            pages_found = set()
            for search_result in debug_info["results"][:3]:  # 상위 3개 결과
                page = search_result.get("page")
                if page and page != "unknown":
                    pages_found.add(str(page))
            
            # 각 페이지에서 관련 이미지 검색
            for page in list(pages_found)[:2]:  # 최대 2개 페이지
                page_images = get_page_images(page, request.message, limit=2)
                images.extend(page_images)
        
        # 이미지가 있으면 답변에 추가 정보 포함
        if images:
            img_info_text = "\n\n📷 관련 이미지:"
            for i, img in enumerate(images[:3]):  # 최대 3개
                img_info_text += f"\n[이미지 {i+1}] 페이지 {img['page']} (관련성: {img['relevance_score']:.2f})"
            answer += img_info_text
        
        # 세션 업데이트
        session["messages"] = result["messages"]
        session["conversation_history"] = result.get("conversation_history", [])
        
        # 메시지 히스토리 제한
        if len(session["messages"]) > 20:
            session["messages"] = session["messages"][-20:]
        
        logger.info("응답 생성 완료")
        
        return ChatResponse(
            answer=answer,
            context=context,
            images=images if images else None,
            debug_info=debug_info if request.debug_mode else None,
            success=True
        )
            
    except Exception as e:
        logger.error(f"채팅 처리 중 오류: {str(e)}")
        return ChatResponse(
            answer="죄송합니다. 서버에서 오류가 발생했습니다.",
            success=False,
            error=str(e)
        )

@app.post("/api/search")
async def search_endpoint(request: SearchRequest):
    """텍스트 검색 엔드포인트"""
    try:
        # 간단한 LangGraph 호출로 검색
        state = {
            "messages": [HumanMessage(content=request.query)],
            "context": "",
            "conversation_history": [],
            "debug_info": {}
        }
        
        result = compiled_graph.invoke(state, config=config)
        debug_info = result.get("debug_info", {})
        
        # 검색 결과 반환
        results = debug_info.get("results", [])
        if request.limit and request.limit < len(results):
            results = results[:request.limit]
        
        return {"results": results, "query": request.query}
        
    except Exception as e:
        logger.error(f"검색 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"검색 오류: {str(e)}")

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "message": "Genesis G80 AI Assistant API is running",
        "active_sessions": len(chat_sessions),
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Genesis G80 AI Assistant API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "chat": "/api/chat",
            "search": "/api/search",
            "reset": "/api/reset"
        }
    }

@app.post("/api/reset")
async def reset_session():
    """대화 세션 초기화"""
    session_id = "default_session"
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    
    # 메모리 정리
    gc.collect()
    
    return {"message": "세션이 초기화되었습니다", "success": True}

# 메모리 정리 미들웨어
@app.middleware("http")
async def clean_memory_after_request(request, call_next):
    """요청 처리 후 메모리 정리"""
    response = await call_next(request)
    
    # 주기적으로 가비지 컬렉션 실행
    if hasattr(clean_memory_after_request, 'request_count'):
        clean_memory_after_request.request_count += 1
    else:
        clean_memory_after_request.request_count = 1
    
    # 10번마다 가비지 컬렉션
    if clean_memory_after_request.request_count % 10 == 0:
        gc.collect()
        logger.info("가비지 컬렉션 실행됨")
    
    return response

# 서버 시작
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Genesis G80 API 서버를 포트 {port}에서 시작합니다...")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    ) 
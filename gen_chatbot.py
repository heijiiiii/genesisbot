# 작성일자 : 2025-04-23

# 1. 필요한 라이브러리 임포트
# 1-1. 환경 변수 라이브러리 임포트
import uuid  # 고유 식별자 생성
import numpy as np  # 수치 연산 모듈
import json  # JSON 파싱용
import re  # 정규표현식 사용
import os  # 운영 체제 관련 함수 임포트
import ast  # 문자열을 파이썬 객체로 변환
from dotenv import load_dotenv  # .env 파일 로드
import sys  # 시스템 관련 함수 임포트
from pathlib import Path  # 경로 처리를 위한 라이브러리

# 1-2. LangChain 라이브러리 임포트
from langchain_cohere import CohereEmbeddings  # Cohere 임베딩 모델
from langchain.schema import Document  # 문서 스키마
from supabase import create_client  # Supabase 클라이언트
from langchain_community.vectorstores.supabase import SupabaseVectorStore  # Supabase 벡터 저장소
from langchain_community.retrievers import BM25Retriever  # BM25 검색기
from langchain_openai import ChatOpenAI  # OpenAI 챗봇 모델

# 1-4. LangGraph 라이브러리 임포트
from typing import Dict, List, Optional, Any, Tuple  # 타입 힌트 임포트
from langchain_core.messages import HumanMessage, AIMessage  # 메시지 타입
from langchain.tools import BaseTool  # 도구 타입
from langgraph.checkpoint.memory import MemorySaver  # 메모리 저장 체크포인터
from langgraph.graph import START, END, MessagesState  # 그래프 상태
from langgraph.graph.state import StateGraph  # 그래프 상태

# 2. 환경 변수 설정
def load_environment():
    """환경 변수를 로드하고 검증합니다."""
    # 로컬 개발 환경에서만 .env.backend 파일 사용
    env_path = Path('.env.backend')
    
    if env_path.exists():
        # 로컬 개발 환경
        load_dotenv(env_path)
        print("✅ 로컬 개발 환경 (.env.backend) 설정을 불러왔습니다.")
    else:
        # 배포 환경 (Supabase 대시보드에서 설정)
        print("ℹ️ 배포 환경: Supabase 대시보드의 환경변수를 사용합니다.")
    
    # 필수 환경 변수 확인
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API 키",
        "COHERE_API_KEY": "Cohere API 키",
        "SUPABASE_URL": "Supabase URL",
        "SUPABASE_SERVICE_ROLE_KEY": "Supabase 서비스 롤 키"
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"- {description} ({var})")
    
    if missing_vars:
        print("\n⚠️ 다음 환경 변수가 설정되지 않았습니다:")
        print("\n".join(missing_vars))
        if env_path.exists():
            print("\n.env.backend 파일에서 환경변수를 확인해주세요.")
        else:
            print("\nSupabase 대시보드에서 환경변수를 확인해주세요.")
        sys.exit(1)
    
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "COHERE_API_KEY": os.getenv("COHERE_API_KEY"),
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_SERVICE_ROLE_KEY": os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
        "PORT": int(os.getenv("PORT", "8000"))
    }

# 환경 변수 로드
env_vars = load_environment()

# Supabase 클라이언트 초기화
try:
    client = create_client(env_vars["SUPABASE_URL"], env_vars["SUPABASE_SERVICE_ROLE_KEY"])
except Exception as e:
    print(f"\n⚠️ Supabase 연결 실패: {str(e)}")
    print("Supabase 연결 정보를 확인해주세요.")
    sys.exit(1)

# 3. 검색기(Retriever) 설정
# 3-1. Cohere 임베딩 (1024차원으로 설정)
try:
    cohere_embeddings = CohereEmbeddings(
        model="embed-v4.0",
        cohere_api_key=env_vars["COHERE_API_KEY"])
except Exception as e:
    print(f"\n⚠️ Cohere 임베딩 모델 초기화 실패: {str(e)}")
    print("Cohere API 키를 확인해주세요.")
    sys.exit(1)

# 3-2. 메타데이터 안전 처리 함수
def safe_get_metadata(item, key, default=None):
    """메타데이터를 안전하게 추출합니다."""
    if not item or not isinstance(item, dict):
        return default
    
    metadata = item.get('metadata', {})
    if not metadata:
        return default
    
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except:
            return default
    
    return metadata.get(key, default)

def ensure_metadata_structure(metadata):
    """메타데이터 구조를 보장합니다."""
    if not metadata or not isinstance(metadata, dict):
        return {
            "page": "unknown",
            "category": "general",
            "section": "main",
            "source": "manual",
            "timestamp": None
        }
    
    # 필수 필드 기본값 설정
    defaults = {
        "page": "unknown",
        "category": "general", 
        "section": "main",
        "source": "manual",
        "timestamp": None
    }
    
    # 기존 값이 있으면 유지, 없으면 기본값 사용
    for key, default_value in defaults.items():
        if key not in metadata or metadata[key] is None:
            metadata[key] = default_value
    
    return metadata

# 3-3. Supabase 벡터 스토어 검색기 정의 (최적화된 버전)
class EnhancedSupabaseRetriever:
    def __init__(self, client, embeddings, table_name="text_embeddings", query_name="match_text_embeddings", k=5):
        self.client = client
        self.embeddings = embeddings
        self.table_name = table_name
        self.query_name = query_name
        self.k = k
    
    def invoke(self, query, page_filter=None):
        try:
            # 1024차원 임베딩 생성
            query_embedding = self.embeddings.embed_query(query)
            
            # RPC 호출로 벡터 검색 수행 (올바른 매개변수 이름 사용)
            rpc_params = {
                "query_embedding": query_embedding,
                "mrch_thresh": 0.3,  # match_threshold -> mrch_thresh
                "mrch_count": self.k  # match_count -> mrch_count
            }
            
            # 페이지 필터가 있으면 추가
            if page_filter:
                rpc_params["page_filter"] = str(page_filter)
            
            matches = self.client.rpc(self.query_name, rpc_params).execute()
            
            docs = []
            if matches.data:
                for i, match in enumerate(matches.data):
                    if 'content' in match and match['content']:
                        # 메타데이터 안전 처리
                        metadata = ensure_metadata_structure(match.get('metadata', {}))
                        metadata['similarity'] = float(match.get('similarity', 0))
                        metadata['source'] = "Vector"
                        metadata['rank'] = i
                        
                        # 페이지 필터 적용
                        if page_filter:
                            page_info = str(metadata.get('page', ''))
                            if page_info != str(page_filter):
                                continue
                        
                        docs.append(Document(
                            page_content=match['content'],
                            metadata=metadata))
            
            return docs
        
        except Exception as e:
            print(f"⚠️ 벡터 검색 오류: {str(e)}")
            return []
    
    def get_relevant_documents(self, query):
        return self.invoke(query)

# 3-4. Supabase 텍스트 벡터 스토어 설정
text_vectorstore = SupabaseVectorStore(
    client=client,
    embedding=cohere_embeddings,
    table_name="text_embeddings",
    query_name="match_text_embeddings")

# 3-5. Supabase 이미지 벡터 스토어 설정
image_vectorstore = SupabaseVectorStore(
    client=client,
    embedding=cohere_embeddings,
    table_name="image_embeddings",
    query_name="match_image_embeddings")

# 3-6. 데이터 로드 및 BM25 검색기 생성
try:
    resp = client.table("text_embeddings").select("content,metadata").execute()
    docs = []
    for item in resp.data:
        metadata = ensure_metadata_structure(item.get("metadata", {}))
        docs.append(Document(
            page_content=item["content"],
            metadata=metadata))
    
    texts = [d.page_content for d in docs]
    bm25 = BM25Retriever.from_texts(
        texts=texts,
        metadatas=[d.metadata for d in docs],
        k=5)
    
    print(f"✅ {len(docs)}개의 문서로 BM25 검색기를 초기화했습니다.")
    
except Exception as e:
    print(f"⚠️ BM25 검색기 초기화 실패: {str(e)}")
    # 빈 검색기로 초기화
    bm25 = BM25Retriever.from_texts(texts=[""], metadatas=[{}], k=5)

# 3-7. 최적화된 벡터 검색기
vector_retriever = EnhancedSupabaseRetriever(
    client=client,
    embeddings=cohere_embeddings,
    table_name="text_embeddings",
    query_name="match_text_embeddings",
    k=5)

# 3-7-2. 이미지 검색기 추가
image_retriever = EnhancedSupabaseRetriever(
    client=client,
    embeddings=cohere_embeddings,
    table_name="image_embeddings",
    query_name="match_image_embeddings",
    k=3)

# 3-8. 하이브리드 검색기 (이미지 검색 포함)
class EnhancedEnsembleRetriever:
    def __init__(self, retrievers: List[Any], weights: Optional[List[float]] = None, verbose: bool = False):
        self.retrievers = retrievers
        if weights is None:
            weights = [1.0 / len(retrievers) for _ in retrievers]
        self.weights = weights
        self.verbose = verbose
        self.retriever_names = ["BM25", "Vector", "Image"]  # 이미지 검색기 추가
    
    def invoke(self, query: str) -> List[Document]:
        all_docs = []
        retriever_docs = {}
        
        for i, retriever in enumerate(self.retrievers):
            try:
                docs = retriever.invoke(query)
                retriever_docs[self.retriever_names[i]] = []
                
                for j, doc in enumerate(docs):
                    if doc.metadata is None:
                        doc.metadata = ensure_metadata_structure({})
                    
                    doc.metadata["source"] = self.retriever_names[i]
                    doc.metadata["original_rank"] = j
                    doc.metadata["retriever_weight"] = float(self.weights[i])
                    
                    if self.retriever_names[i] in ["Vector", "Image"] and "similarity" in doc.metadata:
                        similarity = float(doc.metadata["similarity"])
                        enhanced_score = min(0.95, max(0.1, similarity * 1.2))
                        doc.metadata["score"] = enhanced_score * self.weights[i]
                    else:
                        base_score = 1.0 / (1.0 + j)
                        doc.metadata["score"] = float(base_score * self.weights[i])

                    retriever_docs[self.retriever_names[i]].append(doc)
                        
                all_docs.append(docs)
            
            except Exception as e:
                print(f"⚠️ 검색기 {self.retriever_names[i]} 오류: {str(e)}")
                retriever_docs[self.retriever_names[i]] = []
                all_docs.append([])
                continue
        
        # 결과 통합 및 중복 제거
        seen_contents = set()
        final_docs = []
        
        for docs in all_docs:
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    final_docs.append(doc)
        
        final_docs.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)
        return final_docs[:7]  # 이미지 포함으로 결과 수 증가

# 3-9. 하이브리드 검색기 설정 (이미지 검색 포함)
hybrid_retriever = EnhancedEnsembleRetriever(
    retrievers=[bm25, vector_retriever, image_retriever],
    weights=[0.20, 0.45, 0.35],  # BM25(20%) + Vector(45%) + Image(35%) - 이미지 비중 증가
    verbose=False)

# 4. OpenAI LLM 챗봇 모델 설정
try:
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.2,
        api_key=env_vars["OPENAI_API_KEY"])
    print("✅ OpenAI GPT-4o 모델이 초기화되었습니다.")
except Exception as e:
    print(f"⚠️ OpenAI 모델 초기화 실패: {str(e)}")
    sys.exit(1)

# 5. LangGraph 설정
class AgentState(MessagesState):
    context: str
    conversation_history: Optional[List[Dict]] = None
    debug_info: Optional[Dict] = None

# 6. 검색 도구 정의
class SearchDocumentsTool(BaseTool):
    name: str = "search_documents"
    description: str = "제네시스 G80 매뉴얼에서 관련 정보를 검색합니다."

    def _run(self, query: str) -> Tuple[str, Dict]:
        normalized_query = query.strip().rstrip('.!?')
        
        debug_info = {
            "query": normalized_query,
            "results": [],
            "search_method": "hybrid_retrieval"
        }
        
        try:
            # 하이브리드 검색 수행
            docs = hybrid_retriever.invoke(normalized_query)
            
            if not docs:
                return "매뉴얼에서 관련 정보를 찾을 수 없습니다.", debug_info
            
            # 결과 구성
            result_text = ""
            for i, doc in enumerate(docs):
                metadata = ensure_metadata_structure(doc.metadata)
                
                result_text += f"내용: {doc.page_content}\n"
                result_text += f"카테고리: {metadata.get('category', '없음')}\n"
                result_text += f"페이지: {metadata.get('page', '없음')}\n"
                result_text += f"점수: {metadata.get('score', 0):.3f}\n\n"
                
                # 디버그 정보 저장
                debug_info["results"].append({
                    "rank": i+1,
                    "source": metadata.get("source", "알 수 없음"),
                    "score": float(metadata.get("score", 0)),
                    "page": metadata.get("page", "없음"),
                    "preview": doc.page_content[:100] + "..."
                })
            
            return result_text, debug_info
            
        except Exception as e:
            debug_info["error"] = str(e)
            return f"검색 중 오류가 발생했습니다: {str(e)}", debug_info

# 7. LangGraph 노드 정의
def agent_node_fn(state: AgentState):
    if state.get("conversation_history") is None:
        state["conversation_history"] = []
    
    if state.get("debug_info") is None:
        state["debug_info"] = {}
    
    last_query = state["messages"][-1].content if state["messages"] else ""
    
    if not state.get("context"):
        return {
            "messages": state["messages"],
            "context": None,
            "conversation_history": state["conversation_history"],
            "debug_info": state.get("debug_info", {})
        }

    # 대화 컨텍스트 구성
    conversation_context = ""
    if state["conversation_history"]:
        conversation_context = "이전 대화 내용:\n"
        for i, exchange in enumerate(state["conversation_history"][-3:]):
            conversation_context += f"[대화 {i+1}]\n"
            conversation_context += f"사용자: {exchange['user']}\n"
            if "ai" in exchange:
                conversation_context += f"도우미: {exchange['ai']}\n"
    
    # 프롬프트 구성
    prompt = f"""
    당신은 제네시스 G80 고객을 위한 디지털 컨시어지 어시스턴트입니다.
    고객의 질문에 친절하고 품격 있게 응답하며, 정확하고 신뢰할 수 있는 정보를 제공해주세요.

    {conversation_context}

    참고할 정보:
    {state['context']}

    사용자 질문: {last_query}

    위 정보를 바탕으로 상세하고 친절하게 답변해 주세요.
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        ai_msg = response if isinstance(response, AIMessage) else AIMessage(content=response)
        
        # 대화 이력 업데이트
        state["conversation_history"].append({
            "user": last_query,
            "ai": ai_msg.content
        })
        
        # 대화 이력 제한 (최대 10개)
        if len(state["conversation_history"]) > 10:
            state["conversation_history"] = state["conversation_history"][-10:]
        
        return {
            "messages": state["messages"] + [ai_msg],
            "context": state["context"],
            "conversation_history": state["conversation_history"],
            "debug_info": state.get("debug_info")
        }
    
    except Exception as e:
        error_msg = AIMessage(content=f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}")
        return {
            "messages": state["messages"] + [error_msg],
            "context": state["context"],
            "conversation_history": state["conversation_history"],
            "debug_info": state.get("debug_info")
        }

def search_docs_node(state: AgentState):
    last_query = state["messages"][-1].content if state["messages"] else ""
    search_tool = SearchDocumentsTool()
    result, debug_info = search_tool._run(last_query)
    
    return {
        "messages": state["messages"],
        "context": result,
        "conversation_history": state.get("conversation_history", []),
        "debug_info": debug_info
    }

# 8. LangGraph 워크플로우 설정
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node_fn)
workflow.add_node("search_docs", search_docs_node)

def should_search(state: AgentState):
    return state.get("context") is None

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_search,
    {True: "search_docs", False: END}
)
workflow.add_edge("search_docs", "agent")

# 9. 컴파일 및 초기화
try:
    memory_saver = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=memory_saver)
    thread_id = str(uuid.uuid4())
    state = {"messages": [], "context": "", "conversation_history": [], "debug_info": {}}
    config = {"configurable": {"thread_id": thread_id}}
    print("✅ LangGraph 워크플로우가 초기화되었습니다.")
except Exception as e:
    print(f"⚠️ LangGraph 초기화 실패: {str(e)}")
    sys.exit(1)

# 10. 대화형 인터페이스 실행
if __name__ == "__main__":
    print("\n=== 제네시스 G80 매뉴얼 도우미 ===")
    print("(종료: q 또는 quit)")
    print("(디버그 모드 설정: d 또는 debug)")
    print("(대화 이력 초기화: r 또는 reset)")
    
    debug_mode = False
    available_commands = {
        "q": "종료", "quit": "종료",
        "d": "디버그", "debug": "디버그",
        "r": "초기화", "reset": "초기화"
    }
     
    while True:
        q = input("\n[질문]: ")
        q_lower = q.lower().strip()
        
        if q_lower in available_commands:
            command_type = available_commands[q_lower]
            
            if command_type == "종료":
                print("매뉴얼 도우미를 종료합니다. 좋은 하루 되세요!")
                break
                
            elif command_type == "디버그":
                debug_mode = not debug_mode
                print(f"디버그 모드: {'활성화' if debug_mode else '비활성화'}")
                continue
                
            elif command_type == "초기화":
                state = {"messages": [], "context": "", "conversation_history": [], "debug_info": {}}
                print("대화 이력이 초기화되었습니다.")
                continue
        
        if q_lower.startswith("r") or q_lower.startswith("d") or q_lower.startswith("q"):
            if q_lower not in available_commands:
                print(f"'{q}'은(는) 알 수 없는 명령어입니다. 도움이 필요하시면 질문을 입력해주세요.")
                continue

        state["messages"].append(HumanMessage(content=q))
        try:
            state["context"] = ""
            res = compiled_graph.invoke(state, config=config)
            state = res
            
            last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
            if last_ai:
                print(f"\n[답변]: {last_ai.content}")
                
            if debug_mode and state.get("debug_info"):
                print("\n===== 검색 결과 디버깅 정보 =====")
                debug_info = state["debug_info"]
                
                if "results" in debug_info:
                    print(f"✅ 검색 방법: {debug_info.get('search_method', '알 수 없음')}")
                    print(f"✅ 검색된 결과: {len(debug_info['results'])}개")
                    
                    for result in debug_info["results"]:
                        print(f"[{result['rank']}] [{result['source']}] "
                              f"페이지: {result['page']}, 점수: {result['score']:.3f}")
                        print(f"  미리보기: {result['preview']}")
                
                if "error" in debug_info:
                    print(f"⚠️ 오류: {debug_info['error']}")
                
                print("==================================\n")
                
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            print("죄송합니다. 질문을 처리하는 중 문제가 발생했습니다. 다시 시도해 주세요.")
            
            state = {
                "messages": [HumanMessage(content=q)], 
                "context": "",
                "conversation_history": state.get("conversation_history", []),
                "debug_info": {}
            }
            continue 
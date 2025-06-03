# Genesis G80 AI Assistant 🚗

제네시스 G80 매뉴얼 전용 AI 어시스턴트 - 최적화된 풀스택 솔루션

## ✨ 주요 기능

- **🎨 제네시스 디자인 시스템**: Paperlogy 폰트를 활용한 브랜드 아이덴티티 반영
- **🤖 하이브리드 검색**: BM25(키워드) + Vector(의미) + Image(시각) 통합 검색
- **📱 완전 반응형**: 모바일/태블릿/데스크톱 최적화된 사용자 경험
- **🖼️ 멀티모달 지원**: 매뉴얼 이미지 표시 및 모달 확대 기능
- **💬 실시간 채팅**: 스트리밍 응답 및 로딩 인디케이터
- **📋 메시지 액션**: 복사, 좋아요/싫어요, 피드백 기능
- **🔔 알림 시스템**: 사용자 친화적 토스트 메시지
- **⚡ 성능 최적화**: 폰트 최적화, 이미지 압축, 캐싱 전략
- **📊 모니터링**: 실시간 성능 추적 및 에러 로깅

## 🛠️ 기술 스택

### Frontend (genesis-ui-migration/)
- **Next.js 15** - React 프레임워크 (이미지/폰트 최적화)
- **TypeScript** - 타입 안전성
- **TailwindCSS 4** - 유틸리티 기반 스타일링
- **Paperlogy Font** - 제네시스 전용 폰트
- **Performance Monitoring** - 실시간 성능 추적

### Backend
- **Python LangChain** - AI 워크플로우 프레임워크
- **OpenAI GPT-4o** - AI 모델
- **Cohere Embeddings** - 벡터 검색
- **Supabase** - 데이터베이스 및 이미지 스토리지
- **LangGraph** - 대화 상태 관리
- **Performance Monitoring** - 메모리 및 성능 추적

## 🚀 설치 및 실행

### 1. 환경 설정
```bash
# 환경변수 복사
cp env.example .env

# .env 파일 편집 (필수)
OPENAI_API_KEY=your_openai_key
COHERE_API_KEY=your_cohere_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_key
```

### 2. 백엔드 실행 (API 서버)
```bash
# 의존성 설치
pip install -r requirements.txt

# API 서버 시작 (포트 8000)
python api_server.py
```

### 3. 프론트엔드 실행
```bash
cd genesis-ui-migration

# 의존성 설치
npm install

# 개발 서버 시작 (포트 3000)
npm run dev
```

### 4. 접속 및 테스트
- **프론트엔드**: http://localhost:3000
- **API 문서**: http://localhost:8000/docs
- **API 테스트**: `python test_api_simple.py`

## 📁 최적화된 프로젝트 구조

```
21_genbot/
├── genesis-ui-migration/          # 메인 프론트엔드
│   ├── app/
│   │   ├── api/chat/             # 백엔드 연결 API
│   │   ├── globals.css           # Paperlogy 폰트 + 최적화
│   │   ├── layout.tsx            # 전역 레이아웃
│   │   └── page.tsx              # 메인 페이지 (반응형)
│   ├── components/
│   │   ├── ui/                   # 재사용 UI 컴포넌트
│   │   ├── GenesisChatWidget.tsx # 메인 채팅 위젯
│   │   └── hud-question-cards.tsx # 질문 카드
│   ├── lib/
│   │   ├── monitoring.ts         # 성능 모니터링
│   │   ├── chat-api.ts           # API 연결
│   │   └── utils.ts              # 유틸리티
│   ├── public/font/              # Paperlogy 폰트 파일
│   └── next.config.ts            # 이미지/폰트 최적화
├── backend/
│   └── monitoring.py             # 백엔드 성능 모니터링
├── api_server.py                # 메인 API 서버 (NEW!)
├── gen_chatbot.py               # AI 챗봇 코어 (콘솔 모드)
├── galaxy_chatbot.py            # 레거시 챗봇 (참고용)
├── check_table_structure.py     # DB 구조 확인 도구
├── test_api_simple.py           # API 테스트 도구
├── requirements.txt             # Python 의존성 (최적화됨)
└── delete_folder/               # 정리된 파일들 보관
```

## 🎨 디자인 시스템

### 브랜드 컬러
- **Primary Gold**: `#9D8A68` (제네시스 시그니처)
- **Light Gold**: `#B8A082` (보조 컬러)
- **Background**: Gray gradient (50 → 100)
- **Text Hierarchy**: Gray-900, Gray-800, Gray-600

### 타이포그래피 (Paperlogy)
```css
/* 메인 타이틀 */
Genesis G80: 48px, font-bold, tracking-wide

/* 서브 타이틀 */
QUIET. POWERFUL. ELEGANT: 14px, font-semibold, tracking-wide

/* 본문 */
Body Text: 16px, font-light, tracking-normal
```

### 반응형 브레이크포인트
- **Mobile**: < 768px (글자 크기 조정)
- **Tablet**: 768px - 1024px
- **Desktop**: > 1024px (최대 너비 1200px)

## ⚡ 성능 최적화

### 프론트엔드
- **폰트 최적화**: `font-display: swap`
- **이미지 최적화**: WebP/AVIF 자동 변환
- **캐싱**: 1년 정적 자산 캐시
- **CSS 최적화**: 자동 압축

### 백엔드
- **API 모니터링**: 응답 시간 추적
- **메모리 캐싱**: LRU 캐시 전략
- **리소스 모니터링**: CPU/메모리 사용률 추적

## 📊 모니터링 시스템

### 프론트엔드 모니터링
```typescript
import { PerformanceMonitor } from '@/lib/monitoring';

// 페이지 로드 시간 추적
PerformanceMonitor.logPageView('/home');

// API 호출 모니터링
const response = await monitoredFetch('/api/chat');
```

### 백엔드 모니터링
```python
from backend.monitoring import BackendMonitor

@BackendMonitor.api_timer("chat_endpoint")
def chat_api():
    # API 응답 시간 자동 추적
    pass
```

## 🚦 상태 확인

### 개발 환경
```bash
# 백엔드 상태 확인
curl http://localhost:8000/health

# 프론트엔드 빌드 테스트
cd genesis-ui-migration && npm run build
```

### 성능 지표
- **페이지 로드**: < 2초
- **API 응답**: < 1초 (정상), < 3초 (경고)
- **폰트 로딩**: 즉시 (font-display: swap)

## 🔧 개발 도구

### 코드 품질
```bash
# 린팅
npm run lint

# 타입 체크
npx tsc --noEmit
```

### 성능 분석
```bash
# 번들 분석
npm run build && npm run analyze

# 성능 프로파일링 (개발 환경)
# 브라우저 DevTools → Performance 탭
```

## 🌐 배포

### Render.com 배포
```bash
# 자동 배포 설정됨
git push origin main
```

### 환경변수 (프로덕션)
```env
NODE_ENV=production
OPENAI_API_KEY=prod_key
COHERE_API_KEY=prod_key
SUPABASE_URL=prod_url
SUPABASE_SERVICE_ROLE_KEY=prod_key
```

## 🔗 사용 방법

### 콘솔 챗봇 명령어
- **일반 질문**: 매뉴얼 관련 질문 입력
- **`d` 또는 `debug`**: 디버그 모드 토글
- **`r` 또는 `reset`**: 대화 이력 초기화
- **`q` 또는 `quit`**: 프로그램 종료

### 예시 질문
```
[질문]: G80의 연비는 어떻게 되나요?
[질문]: 타이어 공기압 확인 방법
[질문]: 스마트키 배터리 교체 방법
```

## 📄 라이선스

MIT License

## 🚀 기여하기

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Genesis G80 AI Assistant** - 최고의 성능과 사용자 경험을 제공하는 차세대 매뉴얼 솔루션 ⚡
# CLAUDE.md — H-Walker Graph Web

**새 Claude 세션이 이 레포에 들어왔을 때 먼저 읽어야 할 컨텍스트.**

---

## 👤 사용자

- 조병준 (ARLAB, 케이블 드리븐 보행 재활 로봇 "H-Walker" 연구자)
- 한국어 소통, 직접적이고 간결한 답변 선호
- `python` 아닌 `python3` 사용
- Git 커밋 메시지에 **Claude/AI 흔적 절대 금지** (`Co-Authored-By`, PR 본문 등)
- 로컬 경로: `~/h-walker-graph-web` (main repo) / `~/h-walker-ws` (로봇 코드 별도)
- 다른 프로젝트: `vault` (research-vault), `skiro` (learning capture)

---

## 🎯 이 레포의 정체

**단일 사용자(연구자 본인)용 데스크톱 웹앱** — H-Walker 로부트 CSV 실험 데이터를
자연어로 분석·시각화하고 **논문 Figure를 저널 사이즈 그대로 Export** 하는 워크스페이스.

**핵심 철학: "원툴" — 업로드부터 저널 Export 까지 한 페이지에서 종결.**

사용자가 명시적으로 말한 최우선 요구:

1. 논문 용도 **사이즈까지 완벽** (88.9mm / 181mm 등 정확 치수)
2. 로딩 피드백 필수 (요청 → 즉시 스켈레톤 → 렌더)
3. 여러 그래프를 한 페이지에서 관리 (카드 그리드)
4. 드래그앤드랍 CSV 업로드 **상시 노출**
5. 불필요한 UI 최소화 (`docs/SIMPLIFY.md` 참고)

---

## 🧱 기술 스택

```
Frontend  React 19 + TypeScript + Vite + Zustand (persist)
          - 손으로 쓴 CSS (Tailwind 제거됨)
          - lucide-react 아이콘
          - Pretendard (self-host /fonts) + JetBrains Mono
          - 디자인 원본: design/phase2/core_v3.html (3218줄)

Backend   FastAPI + pandas + matplotlib + Pillow + Anthropic SDK
          - Haiku 4.5 via /api/claude/complete
          - Publication render via matplotlib + bezier path sampler

Launcher  run.py — uvicorn + SPA fallback + static mounts

Vendor    tools/auto_analyzer, tools/graph_analyzer (gait 분석 유틸)
```

---

## 📁 주요 경로

```
h-walker-graph-web/
├── CLAUDE.md                 ← 너 지금 여기
├── README.md                 사용자용 소개
├── SETUP.md                  빈 컴퓨터 설치 가이드
├── requirements.txt          Python deps
├── run.py                    ⭐ 원클릭 런처
├── design/                   ⭐ 원본 디자인 (HANDOFF_CLAUDE_CODE.md 필독!)
│   ├── HANDOFF_CLAUDE_CODE.md   포팅 지침서 (§1~§7)
│   ├── phase2/core_v3.html      기준 목업 3218줄
│   └── fonts/                   Pretendard 전체 weight
├── docs/
│   ├── SIMPLIFY.md           제거/축소/통합 후보 17개
│   └── HANDOVER-*.md
├── frontend/
│   ├── src/
│   │   ├── App.tsx           3-panel 쉘
│   │   ├── store/workspace.ts Zustand (cells/datasets/mode/preset/...)
│   │   ├── data/             verbatim 추출 상수들
│   │   │   ├── journalPresets.ts   6 저널 × 치수/폰트/DPI/팔레트
│   │   │   ├── graphTemplates.ts   8 그래프 × SVG paths
│   │   │   ├── statOps.ts          5 통계 연산
│   │   │   ├── computeMetrics.ts   6 계산 메트릭
│   │   │   ├── canonicalRecipes.ts 데이터셋 타입별 자동 셀 생성
│   │   │   ├── catalogs.ts         HISTORY/STATS_LIB/EXPORT_FORMATS
│   │   │   └── seedCells/seedDatasets
│   │   ├── api/index.ts      fetch 래퍼 (HANDOFF §2 스펙)
│   │   ├── components/
│   │   │   ├── TopNav.tsx        dark/light 로고 자동 스왁
│   │   │   ├── Sidebar.tsx       아이콘 전용, tooltip 팬업
│   │   │   ├── PublicationBar.tsx
│   │   │   ├── Canvas.tsx
│   │   │   ├── DatasetPanel.tsx  드랍존 + recipe 체크박스
│   │   │   ├── cells/
│   │   │   │   ├── Cell.tsx      래퍼
│   │   │   │   ├── GraphCell.tsx ⭐ SVG 렌더 + pub 모드 팔레트 스왁 + Export
│   │   │   │   ├── StatCell.tsx
│   │   │   │   ├── ComputeCell.tsx
│   │   │   │   └── LlmCell.tsx
│   │   │   ├── LlmDock.tsx       하단 입력창 → /api/claude/complete
│   │   │   ├── CmdK.tsx          ⌘K 팬레트 (제거 후보)
│   │   │   ├── FocusOverlay.tsx
│   │   │   ├── ColumnMapperModal.tsx
│   │   │   ├── Drawer.tsx        history/exports/stats/settings
│   │   │   ├── HelpOverlay.tsx   (제거 후보)
│   │   │   └── Toast.tsx
│   │   └── styles/
│   │       ├── app.css           core_v3.html 의 <style> 이식
│   │       └── colors_and_type.css  @font-face + CSS vars
│   ├── public/
│   │   ├── brand/{wordmark,mark}-{dark,light}.svg  ⭐ 2026 리브랜드
│   │   ├── fonts/Pretendard-*.ttf/.otf (19개)
│   │   └── favicon.svg
│   ├── src.legacy/           예전 3-panel UI (gitignored, 참고용만)
│   ├── index.html
│   ├── package.json
│   └── vite.config.ts
├── backend/
│   ├── main.py               FastAPI 앱 팩토리 (run.py 가 대체 사용)
│   ├── routers/
│   │   ├── claude.py         ⭐ Anthropic SDK 직접 호출
│   │   ├── datasets.py       /api/datasets/* (upload/list/mapping)
│   │   ├── graphs.py         ⭐ /api/graphs/render + /bundle (HANDOFF §2.4)
│   │   ├── chat.py / feedback.py / drive.py / journal.py / graph.py   legacy
│   ├── services/
│   │   ├── publication_engine.py  ⭐ JOURNAL_PRESETS (Py mirror) +
│   │   │                            GRAPH_SPECS + bezier sampler + render()
│   │   ├── analysis_engine.py     tools.auto_analyzer 래퍼
│   │   ├── llm_client.py          Claude Haiku + 도메인 지식 + 피드백
│   │   ├── config.py
│   │   ├── knowledge_loader.py / feedback_loader.py / session_state.py
│   │   └── graph_quick.py / graph_publication.py   legacy
│   └── models/schema.py      AnalysisRequest, StatsResult, GraphSpec
├── tools/                    vendored: auto_analyzer + graph_analyzer
└── AppIcon.icns              Mac .app 번들용 (1.6 MB)
```

---

## 🎨 디자인 시스템 (절대 건드리지 말 것)

**팔레트 (유일 source of truth — `frontend/src/styles/app.css` 의 :root):**

| 역할 | 색상 | 용도 |
|---|---|---|
| 배경 | `#0B0E2E` | 전체 앱 |
| 액센트 | `#F09708` | QUICK 토글, 로고, CTA, 헤더 ey |
| 네온 | `#00FFB2` | 상태 인디케이터, stat.sig |
| 보라 | `#A78BFA` | LLM 셀, ASK 프리픽스 |
| 시안 (info) | `#7FB5E4` | COMPUTE 셀, IMU tag |
| Error | `#f87171` | Delete hover, bad p-value |

**타이포:**
- UI: `'Pretendard'` (self-host in /fonts)
- 모노 (JSON, 수치, kbd): `'JetBrains Mono'` (Google Fonts CDN)

**반드시 유지:**
- Glass morphism (nav.topnav, .cell)
- Eyebrow 라벨 (대문자 + letter-spacing .22em)
- `var(--hw-ease)` = `cubic-bezier(.22,1,.36,1)` 모든 전환
- 셀 호버 시 `border-left:3px solid var(--accent)` 뉘앙스

---

## 🔑 HANDOFF §2 엔드포인트 (중요 ⭐)

| 메소드 | 경로 | 상태 | 비고 |
|---|---|---|---|
| POST | `/api/datasets/upload` | ✅ | CSV multipart + pandas 파싱 + column role guess |
| GET | `/api/datasets` / `/{id}` | ✅ | in-memory registry |
| DELETE | `/api/datasets/{id}` | ✅ | 반드시 `Response(status_code=204)` — FastAPI assert 주의 |
| POST | `/api/datasets/{id}/mapping` | ✅ | |
| POST | `/api/graphs/render` | ✅ | ⭐ 저널 사이즈 바이너리 (SVG/PDF/EPS/PNG/TIFF) |
| POST | `/api/graphs/bundle` | ✅ | ZIP + README.txt |
| GET | `/api/graphs/presets` / `/templates` | ✅ | 메타 정보 |
| POST | `/api/claude/complete` | ✅ | Haiku 4.5 직접 호출, 3문장 system prompt |
| GET | `/api/claude/health` | ✅ | key_present 확인 |
| POST | `/api/compute` | ⏳ Phase B | SciPy + pingouin 실제 구현 |
| POST | `/api/stats` | ⏳ Phase B | |
| POST | `/api/export/bundle` | ⏳ Phase B | 통계 포함 ZIP |
| GET | `/api/claude/stream` | ⏳ Phase B | SSE 스트리밍 |

---

## 📐 논문 Export — 검증된 스펙

| 저널 | 1-col mm | 2-col mm | 폰트 | Body pt | Stroke pt | DPI | 팔레트 |
|---|---|---|---|---|---|---|---|
| IEEE | 88.9 | 181 | Times | 8 | 1.0 | 600 | grayscale |
| Nature | 89 | 183 | Helvetica | 7 | 0.5 | 300 | Wong 색맹-safe |
| APA | 85 | 174 | Arial | 10 | 0.75 | 300 | grayscale |
| Elsevier | 90 | 190 (+140 1.5col) | Arial | 8 | 0.5 | 300 | |
| MDPI | 85 | 170 | Palatino | 8 | 0.75 | 1000 | |
| JNER | 85 | 170 | Arial | 8 | 0.75 | 300 | 색맹-safe |

**검증 완료 (smoke test):**
- IEEE col1 SVG → `width="252pt"` = 88.9mm × 2.835 pt/mm ✅
- Nature col2 PNG → 2161×1062 @ 300dpi ✅
- APA PDF 1-page ✅ · MDPI TIFF ✅
- Bundle ZIP (README.txt 포함) ✅

---

## 🚧 아직 안 된 것 (Phase B todo)

`docs/SIMPLIFY.md` 와 HANDOFF §3.3-§3.5 참고. 우선순위 순:

1. `/api/compute` + `/api/stats` 실제 구현 (SciPy, pingouin)
2. `/api/graphs/render` 에 `dataset_id` 실데이터 지원 (현재 GRAPH_SPECS mock)
3. `@dnd-kit/sortable` 로 카드 재배치
4. FocusOverlay brush-zoom (mousedown/move/up)
5. `/api/claude/stream` SSE
6. Playwright e2e (HANDOFF §3.5)
7. **간소화 실행** (`docs/SIMPLIFY.md` Quick wins: 드로어 3개 + cmdK + help 제거)

---

## 🐛 알려진 함정 / Gotchas

1. **`tools/` 는 vendored** — 원래 `~/h-walker-ws/tools/` 에 있던 것. 지우면 안 됨.
2. **FastAPI DELETE 204** — `def delete(...) -> None` + `status_code=204` 는 assert 실패.
   항상 `return Response(status_code=204)` 로.
3. **matplotlib `dashes=None`** — matplotlib 이 `len(None)` 에러. `None` 이면 kwarg 자체 생략.
4. **Vite 의 `/api/*` proxy** — dev 모드에서 localhost:5173 → localhost:8000. vite.config.ts 에 proxy 설정 있음.
5. **Pretendard /fonts** — 폰트 경로가 `url('/fonts/...')` 절대경로. public/fonts/ 에서 서빙.
6. **`hw_graph_dir`** = `~/.hw_graph` (knowledge/feedback/logs/cache). 리셋하려면 이 폴더 삭제.
7. **Zustand persist** — 로컬스토리지 키 `hw_workspace_v1`. 스키마 바뀌면 bump 해야 함.
8. **LlmDock 에러 시 claude.py 직접 호출이라** `/api/claude/health` 로 key_present 먼저 확인.
9. **`npm run dev`** 쓸 때도 **백엔드는 `python3 run.py` 로 별도 실행** 필요.
10. **Legacy 라우터 (`chat.py`, `graph.py` 등)** — 제거 후보지만 현재는 등록돼 있음. 수정 시 Phase 2 쪽과 충돌 주의.

---

## 🔄 일반 작업 흐름 (새 Claude 가 받을 전형적인 요청)

**"XX 버그 고쳐줘" → 체크리스트**
1. 어느 레이어 (frontend/backend/both)? → 해당 파일 먼저 Read
2. `git log --oneline -20` 로 최근 맥락 파악
3. 수정 → `cd frontend && npm run build` 또는 `python3 run.py` 로 실행 검증
4. 커밋 — 메시지는 **"무엇을 왜" 중심, Claude/AI 언급 금지**
5. `git push origin main`

**"새 기능 추가해줘"**
1. `design/HANDOFF_CLAUDE_CODE.md` 에 비슷한 패턴 있는지 확인
2. 기존 컴포넌트 스타일 패턴 따라가기 (특히 `.cell`, `.ds-panel`, `.drawer` 의 eyebrow/label 패턴)
3. store 에 상태 추가 필요하면 `workspace.ts` 에 액션 추가
4. 백엔드 필요하면 HANDOFF §2 스키마 따라서 라우터 생성

**"논문 Export 개선"**
- 치수/폰트/DPI 는 절대로 임의로 바꾸지 말 것. `publication_engine.py` 의 `JOURNAL_PRESETS` 가 single source of truth. 변경 시 `journalPresets.ts` 도 동시 업데이트 (verbatim mirror 유지).
- `GRAPH_SPECS` 는 Phase B 에 실 CSV 데이터로 전환 예정. 지금은 mockup 베지어.

**"채팅이 응답 안 해"**
```bash
curl http://localhost:8000/api/claude/health
# key_present: false → export ANTHROPIC_API_KEY=...
# provider: 'ollama' → config.py 에서 LLM_PROVIDER=anthropic
```

**"프론트엔드 빌드 안 됨"**
- TypeScript strict + noUnusedLocals 켜져 있음. 미사용 변수 정리 필요.
- Tailwind/PostCSS 제거됐으니 그 관련 에러 나오면 `*.config.js` 지워졌는지 확인.

---

## 📌 기억해야 할 선언적 규칙

- **🚫 절대 mock / placeholder 데이터를 UI 에 띄우지 마라.** 사용자 1순위 분노 트리거. 데이터셋 미바인드 상태에서는 빈 상태 (axes only + "Bind a CSV" 메시지) — fake bezier curve / "n strides=14" / "112 spm" 같은 placeholder 절대 금지. 컴포넌트는 항상 실제 백엔드 응답만 렌더해야 한다. 정책 위반 회귀 막는 테스트: `tests/backend/test_no_mock_data.py`.
- **🔁 sync 의 정의 (사용자 명시 2026-04-25)**: 한 sync window = `[rising edge, falling edge]` half-open 구간. **rising edge 가 sync START** (operator 가 버튼 누름 → trial 시작), **falling edge 가 sync END** (operator 가 버튼 뗌 → trial 끝). 한 recording 안에 N 개 window = N 개 trial. 분석은 **window 안의 데이터만** 사용 (window 밖은 prep / rest 구간이라 제외). heel-strike 와 다른 개념. 옛 문서가 "falling-rising-falling 한 cycle" 로 잘못 적었었음 — 정정됨. 구현: `backend/services/sync_align.py:find_sync_windows` / `extract_window_slice` / `align_sources_on_window`, `backend/routers/inspector.py:_detect_sync_windows`, `tools/auto_analyzer/analyzer.py:analyze_file_windows`. 회귀: `tests/backend/test_sync_align.py`, `test_inspector.py`, `test_trial_pairing.py`.
- **삭제된 그래프 (다시 추가 금지 without 명시 동의)**: `peak_box` (asymmetry + per_stride table 로 충분), `debug_ts` (clutter + 줌 안 됨). 사용자가 직접 "다시 넣어줘" 하기 전엔 부활 금지.
- **절대 디자인 팔레트 바꾸지 마라.** `#F09708` / `#00FFB2` / `#0B0E2E` 는 브랜드 정체성.
- **절대 Tailwind 다시 도입하지 마라.** 수동 CSS 로 정리된 상태.
- **절대 이 레포에 Claude/AI 흔적 남기지 마라.** Co-Authored-By 금지.
- **항상 `python3`**, 절대 `python` 쓰지 말 것.
- **커밋 전 `npm run build` 한 번은 돌려봐라.** TypeScript 에러가 CI 없이 잡혀야 함.

---

## 🗺️ 시작 체크리스트 (새 세션용)

1. `README.md` 읽기 (30초)
2. `git log --oneline -15` (현재 상태 파악)
3. 필요시 `design/HANDOFF_CLAUDE_CODE.md` (포팅 지침)
4. 작업 파일의 주변 2~3개 파일 Read
5. 사용자 요청 실행 → 빌드 검증 → 커밋 + 푸시

**막히면:** `git log -p -- path/to/file | head -100` 으로 과거 변경 이유 찾기.

---

*최종 업데이트: 2026-04-22 (Phase 2A + brand logos + publication engine)*

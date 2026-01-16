import streamlit as st
import pandas as pd
import json
from openai import OpenAI
from google import genai
from google.genai import types
from google.api_core import retry

# 기본 분석 규칙 (PROMPT)
DEFAULT_PROMPT = '''[대원칙]

1. 체언, 어간 및 용언만을 수집한다.

2. 순우리말이 아닌, 한자어일 경우에만 반드시 한글(한자) 형태로 표기한다.
한자 표기는 유니코드 한중일 통합 한자를 사용한다.
ex. 주권 → 주권(主權)

3. 맞춤법과 띄어쓰기가 맞지 않을 때, 현대어로 표기한다.

4. "사용자 사전"에 등재된 단어는 그대로 수집하되, 대원칙을 따른다.

[소원칙]

<체언>

1. 고유명사는 모두 붙여서 수집한다.
ex. 국제, 연맹 → 국제연맹(國際聯盟)
    멋진 신세계 → 멋진신세계(新世界) # 소설
    더글라스 맥아더 → 더글라스맥아더
1-1. 고유명사가 한자어인 경우, 대원칙에 따라 한글(한자) 형태로 표기한다.
ex. 장택상 → 장택상(張澤相)

2. 의존명사(것, 바, 중(中), 자(者) 등), 수사는 수집하지 않는다.
단, 한문 숫자 또는 아라비아 숫자 + 명사의 경우 숫자를 삭제하지 않고, 모두 남긴다.
ex. 18세기 → 18세기(世紀)
    1950년 → 1950년(年)
    일국사 → 일국사(一國史)

3. 대명사는 일부 인칭대명사(ex. 오인(吾人), 우리)만 수집하고, 나머지는 수집하지 않는다.

<체언: 복합어(파생어, 합성어)의 경우>

0. 한국어 사전에서 복합어는 '-' 기호를 사용하여 표기한다. 파생어의 경우, 합성어의 경우를 고려한다. 

1. 파생어의 경우,
들, 상(上), 적(的), 꼴, 끼리, 님, 당, 어치, 여, 쯤, 차(次), 하(下), 리(裡) 등의 접미사들은 제거하고 나머지를 남긴다. 이외에는 제거하지 않고 그대로 수집한다.
ex. 씨름꾼(씨름-꾼) → 씨름꾼
    불평등(불-평등) → 불평등(不平等)
    반제국주의(반-제국주의) → 반제국주의(反帝國主義)
    사람들(사람-들) → 사람
예외: 적의 경우,'적' 앞 부분이 한 음절이면 제거하지 않고 그대로 수집한다. ex. 전적, 단적, 목적
    '가급적'은 그대로 수집한다.

2. 합성어의 경우, 한국어 사전에 등재된 단어는 분리하지 않는다.
단, 사전에 미등재된 합성어의 경우 단어를 구성하는 어근 각각의 독립적인 의미가 유지되고 <체언: 명사구의 경우> 1-1, 1-2에 부합한다면 분리한다.
ex. 민주주의(민주-주의) → 민주주의(民主主義)
    자립자영(자립-자영) → 자립자영(自立自營)
    여객역(여객-역) → 여객역(旅客驛)
    대통령후보 → 대통령(大統領), 후보(候補)

3. 파생어와 합성어의 성격을 동시에 가질 경우, 적절히 규칙들을 선택하여 처리한다.

<체언: 명사구의 경우>

0. 한국어 사전에서 한글 맞춤법에 띄어 쓰는 것이 원칙이나 붙여 쓰는 것도 허용한 전문어나 고유 명사는 '^' 기호를 사용하여 표시한다.

1. 다음 두 조건 중 하나를 선택한다.       
1-1. 명사구를 구성하는 단어 사이에 조사를 붙였을 때 그 의미가 동일하거나 유사하다면 서로 분리하고, 어색하다면 그대로 수집한다.
1-2. 명사구를 구성하는 단어 중 어느 하나라도 외부 단어와 대등한 접속이 가능하다면 서로 분리하고, 접속이 불가능하다면 그대로 수집한다.
단, 고유명사인 경우에는 조건 선택 없이 <체언> 1번 규칙을 따른다. "사용자 사전"에 등재된 단어는 조건 선택 없이 대원칙을 따른다. 명사구를 구성하는 단어 중 1음절 단어가 있으면 분리하지 않고, 그대로 수집한다.
ex. 노동문제(노동^문제) = 노동의 문제 → 노동(勞動), 문제(問題) ([인종 및 노동] 문제 → 대등하게 접속 가능)
    국민주권(국민^주권) = 국민의 주권 → 국민(國民), 주권(主權) ([외국인 및 국민] 주권) → 대등하게 접속 가능)

2. 합성어와 명사구의 여부를 판별할 때 띄어쓰기를 고려하지 않는다.

<용언>

1. 대원칙을 고려하면서 ‘-’ 없이 기본형으로 표기한다. 동사 및 형용사만을 수집한다.
단, 2음절 이상 한자어 + (-하다, -되다, -시키다, -답다, -드리다, -받다, -스럽다, -지다)는 '-'를 살려 표기한다.
ex. 뛰다 → 뛰다
    진행하다 → 진행(進行)-하다
    진행되다 → 진행(進行)-되다
    사람답다 → 사람답다
    쾌활하다 → 쾌활(快活)-하다
    대하다 → 대(對)하다

2. 보조 동사, 보조 형용사도 수집한다.'''

# 예시 문장 및 JSON (설명용)
EXAMPLE_SENTENCE_1 = "이것은 이미 수 차 말씀한 바와 같이 일반 인민이 주권을 갖었다고 해서 인민이 주권을 직접 반드시 행사해야 하는 것이 아닙니다."
EXAMPLE_JSON_1 = '''
            
            {
                "1": {
                    "말씀한": {
                        "말씀하다": "명사 / ‘말하다’의 높임말"
                    }
                },
                "2": {
                    "일반": {
                        "일반(一般)": "명사 / 특별하지 아니하고 평범한 수준. 또는 그런 사람들"
                    }
                },
                "3": {
                    "인민이": {
                        "인민(人民)": "명사 / 국가나 사회를 구성하고 있는 사람들"
                    }
                },
                "4": {
                    "주권을": {
                        "주권(主權)": "명사 / 국가의 의사를 최종적으로 결정하는 권력"
                    }
                },
                "5": {
                    "갖었다고": {
                        "가지다": "동사 / 자기 것으로 하다"
                    }
                },
                "6": {
                    "해서": {
                        "하다": "동사 / 사람이나 동물, 물체 따위가 행동이나 작용을 이루다"
                    }
                },
                "7": {
                    "인민이": {
                        "인민(人民)": "명사 / 국가나 사회를 구성하고 있는 사람들"
                    }
                },
                "8": {
                    "주권을": {
                        "주권(主權)": "명사 / 국가의 의사를 최종적으로 결정하는 권력"
                    }
                },
                "9": {
                    "행사해야": {
                        "행사(行使)-하다": "동사 / 권리의 내용을 실현하다"
                    }
                },
                "10": {
                    "하는" : {
                        "하다": "보조 동사 / 앞말의 행동을 시키거나 앞말이 뜻하는 상태가 되도록 함을 나타내는 말"
                    }
                }
            }
            
            '''
EXAMPLE_SENTENCE_2 = "대통령을 민중이 보선한다면 직접 주권을 행사하는 것이 되겠고, 국회에서 선거한다면 간접으로 행사하는 것이 되겠읍니다."
EXAMPLE_JSON_2 =  '''
            
            {
                "1": {
                    "대통령을": {
                        "대통령(大統領)": "명사 / 외국에 대하여 국가를 대표하는 국가의 원수"
                    }
                },
                "2": {
                    "민중이": {
                        "민중(民衆)": "명사 / 국가나 사회를 구성하는 일반 국민"
                    }
                },
                "3": {
                    "보선한다면": {
                        "보선(補選)-하다": "명사 / 보충하여 뽑다"
                    }
                },
                "4": {
                    "주권을": {
                        "주권(主權)": "명사 / 국가의 의사를 최종적으로 결정하는 권력"
                    }
                },
                "5": {
                    "행사하는": {
                        "행사(行使)-하다": "동사 / 권리의 내용을 실현하다"
                    }
                },
                "6": {
                    "되겠고": {
                        "되다": "동사 / 어떤 행위나 일이 일어나거나 행하여지다"
                    }
                },
                "7": {
                    "국회에서": {
                        "국회(國會)": "명사 / 국민의 대표로 구성한 입법 기관"
                    }
                },
                "8": {
                    "선거한다면": {
                        "선거(選擧)-하다": "동사 / 선거권을 가진 사람이 공직에 임할 사람을 투표로 뽑다"
                    }
                },
                "9": {
                    "간접으로": {
                        "간접(間接)": "명사 / 중간에 매개(媒介)가 되는 사람이나 사물 따위를 통하여 맺어지는 관계"
                    }
                },
                "10": {
                    "행사하는" : {
                        "행사(行使)-하다": "동사 / 권리의 내용을 실현하다"
                    }
                },
                "11": {
                    "되겠읍니다" : {
                        "되다": "동사 / 어떤 행위나 일이 일어나거나 행하여지다"
                    }
                }
            }
            
            '''

def init_session_state():
    """
    Session State 변수들을 초기화합니다.
    """
    defaults = {
        'is_running': False,
        'is_paused': False,
        'current_row_index': 0,
        'expanded_data': [],
        'error_sent_ids': [],
        'df_uploaded_cache': None,
        'selected_sentence_column_cache': None,
        'selected_sentence_id_column_cache': None,
        'start_row_cache': None,
        'end_row_cache': None,
        'custom_prompt_dataframe_cache': DEFAULT_PROMPT,
        'gemini_api_key_cache': "",
        'gpt_api_key_cache': "",
        'last_uploaded_file_name': None,
        'user_dictionary_cache': ""
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def pos_analyze(sentence, gemini_api_key, gpt_api_key, model_name, prompt, user_dictionary= "", max_retries=5):
    """
    주어진 문장을 형태소 분석하여 JSON 형식으로 반환합니다.
    """
    final_prompt = prompt
    if user_dictionary and user_dictionary.strip():
        words_list = [w.strip() for w in user_dictionary.split('\n') if w.strip()]
        formatted_dict = ", ".join(words_list)
        final_prompt += f"{formatted_dict}에 있는 단어들은 {final_prompt} 속 [대원칙]에 따라 그대로 수집해 주세요. [소원칙]을 적용하지 마세요."

    if model_name.startswith("gpt") or model_name.startswith("o"):
        if not gpt_api_key:
            st.error("GPT API Key를 입력해주세요.")
            return None
        
        try:
            client = OpenAI(api_key=gpt_api_key)
        
        except Exception as e:
            st.error(f"OpenAI 클라이언트 초기화 오류: {e}")
            return None

        initial_prompts = [
                       {"role": "system", "content": final_prompt},
                       {"role": "user", "content": f"예를 들어, '{EXAMPLE_SENTENCE_1}'의 입력문은 실수 없이 엄밀하게 다음의 json 형식대로 출력해 주세요.\n```json\n{EXAMPLE_JSON_1}\n```"},
                       {"role": "user", "content": f"예를 들어, '{EXAMPLE_SENTENCE_2}'의 입력문은 실수 없이 엄밀하게 다음의 json 형식대로 출력해 주세요.\n```json\n{EXAMPLE_JSON_2}\n```"},
                       {"role": "user", "content": "그럼 다음 홑따옴표 안의 문장을 반드시 위의 json 형태처럼 형태소 분석해 주세요. 실수하거나, 누락하지 마세요."},
                       {"role": "user", "content": f"'{sentence}'"},
                       {"role": "user", "content": "대원칙 1번에 해당되지 않는 단어는 json 기록에서 삭제하고, 그 다음 단어부터 순서대로 번호를 따라 작성하세요."}
                      ]

        
        current_prompts = list(initial_prompts)

        for attempt in range(max_retries + 1):
                def call_gpt_api():
                    return client.chat.completions.create(
                        model= model_name,
                        response_format={"type": "json_object"},
                        messages=current_prompts,
                        reasoning_effort="medium" 
                    )

                response = call_gpt_api()

                if response and response.choices[0].message.content:
                    try:
                        parsed_json = json.loads(response.choices[0].message.content)
                        json_string = response.choices[0].message.content
                        return json_string
                        
                    except json.JSONDecodeError as e:
                        if attempt < max_retries:
                            # 모델에게 오류를 알려주고 수정을 요청하는 프롬프트 추가
                            error_feedback_prompt = [
                                {"role": "user", "content": f"이전 응답이 유효한 JSON 형식이 아니었습니다. 오류: `{str(e)}`"},
                                {"role": "user", "content": f"이 문제를 해결하고 규칙에 따라 다시 정확한 JSON 형식으로만 응답해 주세요."}
                            ]
                            current_prompts.extend(error_feedback_prompt)
                        else:
                            return None
                else:
                    if attempt < max_retries:
                        current_prompts.append(
                            {"role": "user", "content": "응답이 없거나 비어있습니다. 이 문제를 해결하고 다시 정확한 JSON 형식으로 응답해 주세요."}
                        )
                    else:
                        return None            
        

    else:
        if not gemini_api_key:
            st.error("Gemini API Key를 입력해주세요.")
            return None
        
        try:
            client = genai.Client(api_key=gemini_api_key)

        except Exception as e:
            st.error(f"Gemini API 클라이언트 초기화 중 오류 발생: {e}")
            return None

        initial_prompts = [
            final_prompt,
            f"예를 들어, '{EXAMPLE_SENTENCE_1}'의 입력문은 실수 없이 엄밀하게 다음의 json 형식대로 출력해 주세요.\n```json\n{EXAMPLE_JSON_1}\n```",
            f"예를 들어, '{EXAMPLE_SENTENCE_2}'의 입력문은 실수 없이 엄밀하게 다음의 json 형식대로 출력해 주세요.\n```json\n{EXAMPLE_JSON_2}\n```",
            "그럼 다음 홑따옴표 안의 문장을 반드시 위의 json 형태처럼 형태소 분석해 주세요. 실수하거나, 누락하지 마세요.",
            f"'{sentence}'",
            "대원칙 1번에 해당되지 않는 단어는 json 기록에서 삭제하고, 그 다음 단어부터 순서대로 번호를 따라 작성하세요."
        ]

        current_prompts = list(initial_prompts)

        for attempt in range(max_retries + 1):
            # API 호출 (일시적인 오류 시 재시도)
            @retry.Retry(
                predicate=retry.if_transient_error,
                initial=1.0, multiplier=2.0, maximum=10.0, deadline=300.0
            )
            def call_gemini_api():
                return client.models.generate_content(
                    model = model_name,
                    contents=current_prompts,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(thinking_budget=8000)
                    )
                )

            response = call_gemini_api()

            if response and response.text:
                try:
                    parsed_json = json.loads(response.text)
                    json_string = response.text
                    return json_string
                    
                except json.JSONDecodeError as e:
                    if attempt < max_retries:
                        error_feedback_prompt = (
                            f"이전 응답이 유효한 JSON 형식이 아니었습니다. 오류: `{str(e)}`\n"
                            f"이 문제를 해결하고 규칙에 따라 다시 정확한 JSON 형식으로만 응답해 주세요."
                        )
                        current_prompts.append(error_feedback_prompt)
                        st.warning(f"JSON 파싱 오류 발생. 재시도 중... ({attempt + 1}/{max_retries + 1})")
                    else:
                        st.error(f"최대 재시도 횟수를 초과했습니다. JSON 파싱 오류: {e}")
                        return None
            else:
                if attempt < max_retries:
                    current_prompts.append("응답이 없거나 비어있습니다. 이 문제를 해결하고 다시 정확한 JSON 형식으로 응답해 주세요.")
                    st.warning(f"API 응답이 비어있습니다. 재시도 중... ({attempt + 1}/{max_retries + 1})")
                else:
                    st.error("최대 재시도 횟수를 초과했습니다. API 응답이 비어있습니다.")
                    return None
        

def transform(data):
    
    json_data = json.loads(data)

    flattened_data = []
    for key, subdict in json_data.items():
        for subkey, values in subdict.items():
            for term, definition in values.items():
                splitted = definition.split(' / ')
                pos = splitted[0] if len(splitted) > 0 else ""
                definition_text = splitted[-1] if len(splitted) > 1 else ""
                flattened_data.append({
                    'Key': key,
                    'Phrase': subkey,
                    'Term': term,
                    'Pos': pos,
                    'Definition': definition_text
                })

    df = pd.DataFrame(flattened_data)
    return df

# Streamlit 앱 UI
st.set_page_config(layout="wide", page_title="규칙 기반 한국어 문장 분석기 with LLM")

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@700&display=swap'); 

    html, body, [class*="st-"], div[data-testid="stTab"] { /*
        font-family: 'IBM Plex Sans KR', sans-serif !important;
    }

    h1 { 
        font-family: 'IBM Plex Sans KR', sans-serif !important;
        font-weight: 700 !important; /* Make them bold for distinction */
    }

    h2 { 
        font-family: 'IBM Plex Sans KR', sans-serif !important;
        font-weight: 600 !important; /* Make them bold for distinction */
    }

    h3 {
        font-family: 'IBM Plex Sans KR', sans-serif !important;
        font-weight: 600 !important; /* Make them bold for distinction */
    }
    </style>
    """,
    unsafe_allow_html=True
)

init_session_state()

st.title("규칙 기반 한국어 문장 분석기 with LLM")

# 사이드바 설정
st.sidebar.header("설정")

st.sidebar.subheader("LLM 모델 선택")
selected_model_input = st.sidebar.selectbox( 
    "LLM 모델 선택",
    ["gpt-5.2", "o4-mini", "gemini-3-flash-preview", "gemini-3-pro-preview"],
    key="selected_model_cache", 
    help="사용할 모델을 선택하세요."
)

st.sidebar.subheader("API Keys")
selected_model = st.session_state.selected_model_cache

if selected_model.startswith("gpt") or selected_model.startswith("o"):
    gpt_api_key_input = st.sidebar.text_input("GPT API Key", type="password", help="OpenAI API Platform에서 GPT API 키를 발급받아 입력하세요.", value=st.session_state.gpt_api_key_cache)
    if gpt_api_key_input: 
        st.session_state.gpt_api_key_cache = gpt_api_key_input
else:
    gemini_api_key_input = st.sidebar.text_input("Gemini API Key", type="password", help="Google AI Studio에서 Gemini API 키를 발급받아 입력하세요.", value=st.session_state.gemini_api_key_cache)
    if gemini_api_key_input: 
        st.session_state.gemini_api_key_cache = gemini_api_key_input

st.sidebar.markdown("---")
st.sidebar.info(
    """
    이 어플리케이션은 Gemini, GPT (LLM)를 통해 한국어 문장에 대한 규칙 기반 분석을 실시합니다.
     
    사용자의 API 키를 입력하고, 단일 문장 작성 또는 CSV 파일을 업로드하여 분석을 시작할 수 있습니다.
    """
)
st.sidebar.markdown("---")
st.sidebar.info(
    """
    이 플랫폼은 서울대학교 AI연구원 인공지능 디지털인문학 연구센터 연구과제(0670-20250037)의 연구 결과물로서 제작되었습니다.

    이 결과물의 무단 전재 및 재배포를 금합니다. 분석 결과의 사용에 대한 책임은 전적으로 사용자 본인에게 있습니다.
    """
)

# 탭 구성
tab1, tab2 = st.tabs(["단일 문장 분석", "데이터프레임 분석"])

with tab1:
    st.header("단일 문장 분석")
    st.markdown("문장을 입력하여 규칙에 따라 분석합니다.")
    st.markdown("---")
    
    col_input, col_dict = st.columns([2, 1])
    
    with col_input:
        sentence_input = st.text_area(
            "분석할 문장 입력:",
            height=100,
            placeholder="여기에 문장을 입력하세요...",
            help="여기에 분석할 문장을 직접 입력합니다."
        )
    
    with col_dict:
        user_dictionary_single = st.text_area(
            "사용자 사전 (선택사항)",
            height=100,
            placeholder="여기에 단어를 입력하세요...",
            help="여기에 입력된 단어들은 사용자의 의도에 맞게 처리되며, 소원칙이 적용되지 않습니다. 줄바꿈으로 단어를 구분하세요."
        )

    with st.expander("분석 규칙 (PROMPT)", expanded=False):
        st.text_area(
            "분석 규칙",
            DEFAULT_PROMPT,
            height=400,
            help="모델이 분석을 수행할 때 따르는 기본 규칙입니다."
        )

    if st.button("분석 시작 / 다시 시작", key="single_analyze_button"):
        # 모델에 따른 API 키 확인
        api_key_valid = False
        if st.session_state.selected_model_cache.startswith("gpt") or st.session_state.selected_model_cache.startswith("o"):
             if st.session_state.gpt_api_key_cache:
                 api_key_valid = True
             else:
                 st.warning("GPT 모델을 사용하려면 OpenAI GPT API Key가 필요합니다.")
        else:
             if st.session_state.gemini_api_key_cache:
                 api_key_valid = True
             else:
                 st.warning("Gemini 모델을 사용하려면 Google Gemini API Key가 필요합니다.")

        if api_key_valid:
            if not sentence_input.strip():
                st.warning("분석할 문장을 입력해주세요.")
            else:
                with st.spinner(f"{st.session_state.selected_model_cache} 모델로 문장 분석 중..."):
                    json_output = pos_analyze(
                        sentence_input, 
                        st.session_state.gemini_api_key_cache, 
                        st.session_state.gpt_api_key_cache,
                        st.session_state.selected_model_cache, 
                        prompt=DEFAULT_PROMPT,
                        user_dictionary=user_dictionary_single 
                    ) 

                if json_output:
                    df_result = transform(json_output)
                    result_view_tab1, result_view_tab2 = st.tabs(["분석 결과(Dataframe)", "JSON 원본 데이터"])
                    # 탭 1: 데이터프레임
                    with result_view_tab1:
                        if not df_result.empty:
                            # use_container_width=True로 가로 폭을 꽉 채움
                            st.dataframe(df_result, use_container_width=True)
                            
                            st.download_button( 
                                label="분석 결과 CSV 다운로드", 
                                data=df_result.to_csv(index=False).encode('utf-8-sig'), 
                                file_name="analyzed_dataframe.csv", 
                                mime="text/csv", 
                                key="download_single_csv"
                            )
                        else:
                            st.info("변환된 데이터프레임이 비어 있습니다. JSON 결과가 규칙에 맞지 않을 수 있습니다.")

                    # 탭 2: JSON 원본 (검증용)
                    with result_view_tab2:
                        st.json(json_output)
                else:
                    st.error("문장 분석에 실패했습니다. API 키, 모델 또는 규칙을 확인해주세요.")

with tab2:
    st.header("데이터프레임 분석")
    st.markdown("CSV 파일을 업로드하여 지정된 열에 있는 문장들을 일괄적으로 분석합니다.")
    st.markdown("**CSV 파일은 반드시 문장 번호가 담긴 column을 포함해야 합니다.**")
    st.markdown("---")
    
    with st.expander("분석 설정 및 규칙 (사용자 사전 포함)", expanded=False):
        st.text_area(
            "분석 규칙 (PROMPT)",
            DEFAULT_PROMPT, 
            height=400,
            key="dataframe_prompt_display",
            help="모델이 분석을 수행할 때 따르는 기본 규칙입니다."
        )
        
        user_dictionary_dataframe_input = st.text_area(
            "사용자 사전 (선택사항)",
            st.session_state.user_dictionary_cache,
            height=150,
            key="dataframe_user_dict_input",
            placeholder="여기에 단어를 입력하세요...",
            help="여기에 입력된 단어들은 사용자의 의도에 맞게 처리되며, 소원칙이 적용되지 않습니다. 줄바꿈으로 단어를 구분하세요."
        )
        if user_dictionary_dataframe_input:
             st.session_state.user_dictionary_cache = user_dictionary_dataframe_input


    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"], help="분석할 문장이 포함된 CSV 파일을 업로드하세요.")
    df_preview_placeholder = st.empty()

    if uploaded_file is not None:
        try:
            # Fix #5: 파일 교체 시 관련 컬럼 Cache 및 상태 완전 초기화
            if st.session_state.df_uploaded_cache is None or uploaded_file.name != st.session_state.last_uploaded_file_name: 
                st.session_state.df_uploaded_cache = pd.read_csv(uploaded_file) 
                st.session_state.last_uploaded_file_name = uploaded_file.name 
                # 상태 및 캐시 명시적 초기화
                st.session_state.is_running = False
                st.session_state.is_paused = False
                st.session_state.current_row_index = 0
                st.session_state.expanded_data = []
                st.session_state.error_sent_ids = []
                st.session_state.selected_sentence_column_cache = None
                st.session_state.selected_sentence_id_column_cache = None
                st.session_state.start_row_cache = 1
                st.session_state.end_row_cache = len(st.session_state.df_uploaded_cache)

            df_preview_placeholder.subheader("업로드된 데이터 미리보기:")
            df_preview_placeholder.dataframe(st.session_state.df_uploaded_cache.head(10)) 

            column_options = st.session_state.df_uploaded_cache.columns.tolist()

            # 컬럼 선택 로직
            default_sent_idx = 0
            if st.session_state.selected_sentence_column_cache in column_options:
                default_sent_idx = column_options.index(st.session_state.selected_sentence_column_cache)
            elif 'sent_raw' in column_options:
                default_sent_idx = column_options.index('sent_raw')
            
            selected_sentence_column_input = st.selectbox("문장 column 선택", options=column_options, index=default_sent_idx)
            if selected_sentence_column_input: st.session_state.selected_sentence_column_cache = selected_sentence_column_input

            default_id_idx = 0
            if st.session_state.selected_sentence_id_column_cache in column_options:
                 default_id_idx = column_options.index(st.session_state.selected_sentence_id_column_cache)
            elif 'sent_id' in column_options:
                default_id_idx = column_options.index('sent_id')

            selected_sentence_id_column_input = st.selectbox("문장 번호 column 선택", options=column_options, index=default_id_idx) 
            if selected_sentence_id_column_input: st.session_state.selected_sentence_id_column_cache = selected_sentence_id_column_input

            col1, col2 = st.columns(2)
            with col1:
                start_row_input = st.number_input("시작 행 번호", min_value=1, max_value=len(st.session_state.df_uploaded_cache), value=st.session_state.start_row_cache or 1)
                if start_row_input: st.session_state.start_row_cache = start_row_input 
            with col2:
                end_row_input = st.number_input("끝 행 번호", min_value=1, max_value=len(st.session_state.df_uploaded_cache), value=st.session_state.end_row_cache or len(st.session_state.df_uploaded_cache))
                if end_row_input: st.session_state.end_row_cache = end_row_input 
             
            if st.session_state.start_row_cache > st.session_state.end_row_cache: 
                st.error("시작 행 번호가 끝 행 번호보다 큽니다.")
                st.stop()

            # --- 기능 복원 구간: 일시정지, 재개, 중단 ---
            col_buttons = st.columns(3)
            
            # 1. 시작 / 재개 버튼
            with col_buttons[0]:
                if st.button("분석 시작 / 다시 시작", key="start_resume_button", disabled=st.session_state.is_running and not st.session_state.is_paused):
                    valid_key = False
                    if st.session_state.selected_model_cache.startswith("gpt") or st.session_state.selected_model_cache.startswith("o"):
                        if st.session_state.gpt_api_key_cache: valid_key = True
                        else: st.warning("GPT 모델을 사용하려면 OpenAI GPT API Key가 필요합니다.")
                    else:
                        if st.session_state.gemini_api_key_cache: valid_key = True
                        else: st.warning("Gemini 모델을 사용하려면 Google Gemini API Key가 필요합니다.")

                    if valid_key:
                        st.session_state.is_running = True
                        st.session_state.is_paused = False
                        
                        # 범위 계산
                        range_len = st.session_state.end_row_cache - st.session_state.start_row_cache + 1
                        # 새로 시작하는 경우 (처음이거나 완료 후 다시 시작)
                        if st.session_state.current_row_index == 0 or st.session_state.current_row_index >= range_len:
                            st.session_state.current_row_index = 0
                            st.session_state.expanded_data = [] # 데이터 초기화
                            st.session_state.error_sent_ids = []
                        
                        st.rerun()

            # 2. 일시정지 버튼
            with col_buttons[1]:
                 if st.button("일시정지", key="pause_button", disabled=not st.session_state.is_running or st.session_state.is_paused):
                    st.session_state.is_paused = True
                    st.rerun()

            # 3. 중단 후 초기화 버튼
            with col_buttons[2]:
                if st.button("중단 후 처음부터", key="stop_restart_button", disabled=not st.session_state.is_running and not st.session_state.is_paused):
                    st.session_state.is_running = False
                    st.session_state.is_paused = False
                    st.session_state.current_row_index = 0
                    st.session_state.expanded_data = []
                    st.session_state.error_sent_ids = []
                    st.rerun()

            # 실행 로직 (일시정지/재개를 위해 st.rerun 활용 패턴 복원)
            if st.session_state.is_running and not st.session_state.is_paused:
                df_subset = st.session_state.df_uploaded_cache.iloc[st.session_state.start_row_cache-1 : st.session_state.end_row_cache]
                total_rows = len(df_subset)
                
                # 현재 인덱스가 범위 내에 있으면 처리
                if st.session_state.current_row_index < total_rows:
                    # 실제 인덱스 추출
                    original_idx = df_subset.index[st.session_state.current_row_index]
                    row = df_subset.loc[original_idx]
                    
                    # 진행률 표시
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    current_progress = (st.session_state.current_row_index + 1) / total_rows
                    progress_bar.progress(current_progress)
                    progress_text.text(f"{st.session_state.selected_model_cache} 모델로 {st.session_state.current_row_index + 1}/{total_rows} 문장 처리 중...")

                    try:
                        sentence = str(row[st.session_state.selected_sentence_column_cache])
                        sent_id_val = row.get(st.session_state.selected_sentence_id_column_cache, "Unknown")
                        
                        pos_result = pos_analyze(
                            sentence, 
                            st.session_state.gemini_api_key_cache, 
                            st.session_state.gpt_api_key_cache,
                            st.session_state.selected_model_cache, 
                            prompt=DEFAULT_PROMPT,
                            user_dictionary=st.session_state.user_dictionary_cache
                        )
                        
                        if pos_result:
                            transformed_df = transform(pos_result)
                            if not transformed_df.empty:
                                transformed_df['sent_id'] = sent_id_val
                                st.session_state.expanded_data.append(transformed_df)
                            else:
                                st.session_state.error_sent_ids.append({'sent_id': sent_id_val, 'error_message': '데이터 비어있음'})
                        else:
                            st.session_state.error_sent_ids.append({'sent_id': sent_id_val, 'error_message': '분석 실패(None)'})
                            
                    except Exception as e:
                        sent_id_val = row.get(st.session_state.selected_sentence_id_column_cache, "Unknown")
                        st.session_state.error_sent_ids.append({'sent_id': sent_id_val, 'error_message': str(e)})

                    # 다음 인덱스로 이동 후 Rerun (이것이 일시정지를 가능하게 함)
                    st.session_state.current_row_index += 1
                    st.rerun()

                else:
                    # 완료 시
                    st.session_state.is_running = False
                    st.session_state.is_paused = False
                    st.rerun()

            # 결과 표시 (분석 완료 후 세션 데이터 표시)
            if st.session_state.expanded_data: 
                st.subheader("분석 결과") 
                result_df = pd.concat(st.session_state.expanded_data, ignore_index=True) 
                st.dataframe(result_df) 
                st.download_button( 
                    label="분석 결과 CSV 다운로드", 
                    data=result_df.to_csv(index=False).encode('utf-8-sig'), 
                    file_name="analyzed_dataframe.csv", 
                    mime="text/csv"
                )
                
            if st.session_state.error_sent_ids: 
                st.subheader("처리 오류 목록") 
                st.warning(f"{len(st.session_state.error_sent_ids)}개의 문장 처리 중 오류가 발생했습니다.") 
                error_df = pd.DataFrame(st.session_state.error_sent_ids) 
                st.dataframe(error_df) 
                st.download_button( 
                    label="오류 목록 CSV 다운로드", 
                    data=error_df.to_csv(index=False).encode('utf-8-sig'), 
                    file_name="analysis_errors.csv", 
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"처리 중 오류 발생: {e}")
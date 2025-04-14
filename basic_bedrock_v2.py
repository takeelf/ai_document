from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
import json

def get_system_prompt() -> str:
    return """
    당신은 개인 자산 정보(계좌, 주식, 대출, 연소득 등)를 부동산과 함께 분석해주는 부동산·자산 컨설턴트 역할을 수행하는 AI 어시스턴트입니다.

사용자의 자산 정보는 다음의 자산문맥을 기반으로 합니다
자산문맥:
{finance_mydata}

사용자가 원하는 부동산 정보는 다음의 부동산문맥을 기반으로 합니다
부동산문맥:
{realestate}

이 정보를 바탕으로 다음의 솔루션을 제시합니다
   - 사용자의 개인자산에 따른 대상 지역 부동산 상황 (대출은 얼마를 더 받아야 하는지, 보통 어느정도 이자로 얼마의 기간동안 갚으면서 살아야 하는지)
   - 연소득에 따른 이자와 원금을 지불하고 나면 얼마의 생활비로 살아야 하는지 분석해주세요
   - 생활비를 분석할때는 의류, 식비, 교통비로 나눠서 재미있게 분석해주세요 (당신은 생활비 부족으로 3만원 이상의 옷은 못삽니다 혹은 탕수육은 한달에 한번씩만 먹어야 합니다 등등)
   - 문장을 짜르지 않고 천천히 생각해서 끝까지 작성해주세요

당신의 응답은 반드시 다음 JSON 구조로 제공해야 합니다:
```json
{{
    "current_asset_situation": "사용자의 현재 자산 상황에 대한 분석 내용...",
    "real_estate_purchase_analysis": "부동산 구매에 관한 분석 내용...",
    "loan_scenario": "필요한 대출 금액 및 상환 계획...",
    "monthly_living_expenses_analysis": {{
        "living_expense_advice": "유머있고 재미있는 전반적인 생활비 조언...",
        "clothing_expenses": "유머있고 재미있는 의류비 관련 분석 및 조언...",
        "food_expenses": "유머있고 재미있는 식비 관련 분석 및 조언...",
        "transportation_expenses": "유머있고 재미있는 교통비 관련 분석 및 조언..."
    }},
    "realistic_advice": "전체적인 재정 상황에 대한 현실적인 조언..."
}}

반드시 위 JSON 형식에 맞게 응답해야 하며, 각 필드에 상세하고 유용한 내용을 작성해주세요. 기타 정책(금지·제한된 행동 등)은 AWS Bedrock 에이전트 기본 정책을 준수합니다.
"""

def get_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", get_system_prompt()),
        ("human", "내 자산정보와 원하는 아파트 부동산정보를 바탕으로 얼마나 이자와 원금을 계산하여 알려주고 얼마의 생활비로 살아야 하는지 분석해주세요. 생활비를 분석할때는 의류, 식비, 교통비로 나눠서 재미있게 분석해주세요 (당신은 생활비 부족으로 3만원 이상의 옷은 못삽니다 혹은 탕수육은 한달에 한번씩만 먹어야 합니다 등등) JSON 형식으로 응답해주세요."),
    ])


def run_llm_model(tone: str) -> str:
    llm = ChatBedrock(
        model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        temperature=0,
        region_name = "us-west-2",
        model_kwargs = {
            "max_tokens": 9999  
        }
    )
    # tone = get_response_tone(tone)
    chain = get_prompt() | llm | JsonOutputParser()
    response = chain.invoke({"finance_mydata": get_finance_mydata(), "realestate": get_realestate()})
    return json.dumps(response, ensure_ascii=False, indent=4)

def get_finance_mydata() -> str:
    return """
{
  "credit_information": {
    "credit_score": 750,
    "credit_history": {
      "delinquency_records": 0,
      "loan_history": [
        {
          "loan_type": "personal_loan",
          "amount": 20000000,
          "status": "repaid"
        }
      ]
    },
    "credit_card_usage": {
      "total_credit_limit": 15000000,
      "current_balance": 3000000
    }
  },
  "income_and_employment": {
    "annual_income": 60000000,
    "monthly_salary": 5000000,
    "employment_status": "full-time",
    "employer": "Company A"
  },
  "liquid_assets": {
    "savings_account_balance": 20000000,
    "average_monthly_balance": 10000000,
    "average_monthly_deposits": 5500000,
    "average_monthly_withdrawals": 4500000,
    "other_liquid_assets": {
      "gold": 10000000,
      "foreign_currency": 5000000,
      "cash_equivalents": 7000000
    }
  },
  "illiquid_assets": {
    "real_estate": [
      {
        "address": "서울특별시 강남구 테헤란로 123",
        "market_value": 800000000,
        "mortgage_balance": 250000000
      }
    ],
    "other_illiquid_assets": {
      "artworks": 15000000
    }
  },
  "investment_information": {
    "stocks": 30000000,
    "bonds": 20000000,
    "retirement_savings": 80000000
  },
  "loans_and_debts": {
    "existing_loans": [
      {
        "loan_type": "mortgage",
        "original_amount": 300000000,
        "current_balance": 250000000,
        "interest_rate": 3.5,
        "monthly_payment": 1500000
      }
    ],
    "credit_card_debt": 3000000,
    "other_debts": 0
  },
  "spending_patterns": {
    "fixed_expenses_total": 2000000,
    "variable_expenses_last_3_months": {
      "groceries": {
        "month_1": 500000,
        "month_2": 480000,
        "month_3": 520000
      },
      "transportation": {
        "month_1": 300000,
        "month_2": 320000,
        "month_3": 280000
      },
      "entertainment": {
        "month_1": 200000,
        "month_2": 250000,
        "month_3": 180000
      },
      "dining_out": {
        "month_1": 400000,
        "month_2": 380000,
        "month_3": 450000
      }
    }
  }
}

"""

def get_finance_mydata2() -> str:
    return """
    자산목록:
    - 계좌
        우리은행 : 41567000
        케이뱅크 : 80000000

    - 대출
        국민은행 : 30000000

    - 연소득: 50000000

    - 주식
        삼성전자 : 3000000
        카카오 : 2000000
        현대자동차: 10000000

    - 부동산 : 700000000
    """

def get_realestate() -> str:
    return """
동천동 래미안 : 1100000000    
    """

def get_realestate2() -> str:
    return """
래미안 대치 팰리스 : 3500000000    
    """

def get_response_tone(tone: str) -> str:
    if tone == "strict":
        return "강하게 혼내는 말투"
    elif tone == "weak":
        return "유하게 달래는 말투"
    else:
        return "보통 말투"

if __name__ == "__main__":
    response = run_llm_model("normal")
    print(response) 



import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def simple_df():
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "No", "Yes"],
            "overtime": ["Yes", "No", "Yes", "No"],
            "monthly_income": [3000.0, 5000.0, 4000.0, 6000.0],
            "job_satisfaction": [1, 3, 2, 4],
        }
    )


# --- attrition_rate ---

def test_attrition_rate_returns_expected_percent(simple_df):
    assert attrition_rate(simple_df) == 50.0


def test_attrition_rate_all_stayed():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


def test_attrition_rate_all_left():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


# --- attrition_by_department ---

def test_attrition_by_department_returns_expected_columns(simple_df):
    result = attrition_by_department(simple_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_correct_rates(simple_df):
    result = attrition_by_department(simple_df)
    sales = result[result["department"] == "Sales"].iloc[0]
    hr = result[result["department"] == "HR"].iloc[0]
    assert sales["employees"] == 2
    assert sales["leavers"] == 1
    assert sales["attrition_rate"] == 50.0
    assert hr["employees"] == 2
    assert hr["leavers"] == 1
    assert hr["attrition_rate"] == 50.0


def test_attrition_by_department_sorted_descending():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5],
            "department": ["Sales", "Sales", "HR", "HR", "HR"],
            "attrition": ["Yes", "Yes", "Yes", "No", "No"],
        }
    )
    result = attrition_by_department(df)
    rates = result["attrition_rate"].tolist()
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_returns_expected_columns(simple_df):
    result = attrition_by_overtime(simple_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_correct_rates():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "overtime": ["Yes", "Yes", "No", "No"],
            "attrition": ["Yes", "Yes", "No", "No"],
        }
    )
    result = attrition_by_overtime(df)
    yes_row = result[result["overtime"] == "Yes"].iloc[0]
    no_row = result[result["overtime"] == "No"].iloc[0]
    assert yes_row["attrition_rate"] == 100.0
    assert no_row["attrition_rate"] == 0.0


# --- average_income_by_attrition ---

def test_average_income_by_attrition_returns_expected_columns(simple_df):
    result = average_income_by_attrition(simple_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_correct_values():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "attrition": ["Yes", "Yes", "No", "No"],
            "monthly_income": [3000.0, 5000.0, 7000.0, 9000.0],
        }
    )
    result = average_income_by_attrition(df)
    yes_avg = result[result["attrition"] == "Yes"]["avg_monthly_income"].iloc[0]
    no_avg = result[result["attrition"] == "No"]["avg_monthly_income"].iloc[0]
    assert yes_avg == 4000.0
    assert no_avg == 8000.0


# --- satisfaction_summary ---

def test_satisfaction_summary_returns_expected_columns(simple_df):
    result = satisfaction_summary(simple_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_rate_uses_group_headcount_not_total_leavers():
    # 4 employees at satisfaction=2, 2 leave → rate should be 50%, not 100%
    # (total leavers = 2, so wrong denominator would give 2/2 = 100%)
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "job_satisfaction": [2, 2, 2, 2],
            "attrition": ["Yes", "Yes", "No", "No"],
        }
    )
    result = satisfaction_summary(df)
    rate = result[result["job_satisfaction"] == 2]["attrition_rate"].iloc[0]
    assert rate == 50.0


def test_satisfaction_summary_sorted_by_satisfaction():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "job_satisfaction": [3, 1, 4, 2],
            "attrition": ["No", "Yes", "No", "Yes"],
        }
    )
    result = satisfaction_summary(df)
    levels = result["job_satisfaction"].tolist()
    assert levels == sorted(levels)


def test_satisfaction_summary_correct_rates():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "job_satisfaction": [1, 1, 2, 2, 2, 2],
            "attrition": ["Yes", "Yes", "Yes", "No", "No", "No"],
        }
    )
    result = satisfaction_summary(df)
    rate_1 = result[result["job_satisfaction"] == 1]["attrition_rate"].iloc[0]
    rate_2 = result[result["job_satisfaction"] == 2]["attrition_rate"].iloc[0]
    assert rate_1 == 100.0
    assert rate_2 == 25.0

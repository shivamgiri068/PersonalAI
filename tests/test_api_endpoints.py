import pytest
from fastapi.testclient import TestClient
from backend.app.database import init_db
from backend.app.main import app

# Ensure database tables exist for test client
init_db()
client = TestClient(app)

def test_root_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "project" in data

def test_openapi_docs():
    response = client.get("/docs")
    assert response.status_code == 200

def test_get_and_update_profile():
    get_res = client.get("/api/profile")
    assert get_res.status_code == 200
    data = get_res.json()
    assert "name" in data

    update_payload = {
        "name": "Shivam Giri",
        "education": "B.Tech CSE",
        "skills": "Python, FastAPI, RAG, FAISS, SQL",
        "interests": "Generative AI Engineering",
        "response_style": "Concise and structured"
    }
    put_res = client.put("/api/profile", json=update_payload)
    assert put_res.status_code == 200
    updated_data = put_res.json()
    assert updated_data["name"] == "Shivam Giri"

def test_list_documents_and_stats():
    res_list = client.get("/api/documents")
    assert res_list.status_code == 200

    res_stats = client.get("/api/documents/stats")
    assert res_stats.status_code == 200
    stats = res_stats.json()
    assert "total_documents" in stats

def test_chat_endpoint():
    payload = {
        "message": "What skills are in my profile?",
        "use_rag": True
    }
    res = client.post("/api/chats", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "conversation_id" in data
    assert "content" in data
    assert data["role"] == "assistant"

def test_job_analyzer_endpoint():
    payload = {
        "job_description": "Hiring Python Generative AI Engineer with experience in FastAPI, LangChain, FAISS, and SQL."
    }
    res = client.post("/api/job/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "matching_skills" in data
    assert "preparation_topics" in data

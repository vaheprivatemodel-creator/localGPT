import json
import http.server
import socketserver
import cgi
import os
import uuid
from urllib.parse import urlparse, parse_qs
import requests  # 🆕 Import requests for making HTTP calls
import sys
from datetime import datetime

# Add parent directory to path so we can import rag_system modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RAG system modules for complete metadata
try:
    from rag_system.main import PIPELINE_CONFIGS
    RAG_SYSTEM_AVAILABLE = True
    print("✅ RAG system modules accessible from backend")
except ImportError as e:
    PIPELINE_CONFIGS = {}
    RAG_SYSTEM_AVAILABLE = False
    print(f"⚠️ RAG system modules not available: {e}")

from ollama_client import OllamaClient
from database import db, generate_session_title, verify_password, hash_password
import simple_pdf_processor as pdf_module
from simple_pdf_processor import initialize_simple_pdf_processor
from typing import List, Dict, Any
import re

# 🆕 Reusable TCPServer with address reuse enabled
class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

class ChatHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.ollama_client = OllamaClient()
        super().__init__(*args, **kwargs)
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Access-Control-Allow-Credentials', 'true')
        self.end_headers()

    # ── Auth helpers ─────────────────────────────────────────────

    def _get_current_user(self):
        """Extract Bearer token (header or ?token= query param) and return the user dict, or None."""
        # 1. Try Authorization header
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
            if token:
                return db.get_user_by_token(token)
        # 2. Fall back to ?token= query param (needed for direct browser download links)
        qs = parse_qs(urlparse(self.path).query)
        token = qs.get("token", [None])[0]
        if token:
            return db.get_user_by_token(token)
        return None

    def _require_auth(self):
        """Return user dict or send 401 and return None."""
        user = self._get_current_user()
        if not user:
            self.send_json_response({"error": "Unauthorized – please log in"}, status_code=401)
        return user

    def _require_admin(self):
        """Return user dict if admin, else send 403 and return None."""
        user = self._require_auth()
        if user and user.get("role") != "admin":
            self.send_json_response({"error": "Forbidden – admin only"}, status_code=403)
            return None
        return user
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)

        # ── Public ────────────────────────────────────────────────
        if parsed_path.path == '/health':
            self.send_json_response({
                "status": "ok",
                "ollama_running": self.ollama_client.is_ollama_running(),
                "available_models": self.ollama_client.list_models(),
                "database_stats": db.get_stats()
            })
            return
        if parsed_path.path == '/auth/me':
            user = self._require_auth()
            if user:
                self.send_json_response({
                    "id": user["id"], "email": user["email"],
                    "name": user["name"], "role": user["role"]
                })
            return

        # ── All other routes require auth ─────────────────────────
        user = self._require_auth()
        if not user:
            return

        if parsed_path.path == '/sessions':
            self.handle_get_sessions(user)
        elif parsed_path.path == '/sessions/cleanup':
            self.handle_cleanup_sessions()
        elif parsed_path.path == '/models':
            self.handle_get_models()
        elif parsed_path.path == '/indexes':
            self.handle_get_indexes()
        elif parsed_path.path == '/admin/users':
            self.handle_admin_list_users(user)
        elif parsed_path.path.startswith('/indexes/') and parsed_path.path.count('/') == 2:
            index_id = parsed_path.path.split('/')[-1]
            self.handle_get_index(index_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/documents'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_get_session_documents(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/indexes'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_get_session_indexes(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.count('/') == 2:
            session_id = parsed_path.path.split('/')[-1]
            self.handle_get_session(session_id)
        elif parsed_path.path in ('/audit', '/audit/'):
            self.handle_get_audit_log()
        elif parsed_path.path == '/audit/export':
            self.handle_export_audit_log()
        elif parsed_path.path.startswith('/audit/') and parsed_path.path.count('/') == 2:
            audit_id = parsed_path.path.split('/')[-1]
            self.handle_get_audit_entry(audit_id)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)

        # ── Public auth endpoints ─────────────────────────────────
        if parsed_path.path == '/auth/login':
            self.handle_auth_login()
            return
        if parsed_path.path == '/auth/logout':
            self.handle_auth_logout()
            return

        # ── All other POST routes require auth ────────────────────
        user = self._require_auth()
        if not user:
            return

        if parsed_path.path == '/chat':
            self.handle_chat()
        elif parsed_path.path == '/sessions':
            self.handle_create_session(user)
        elif parsed_path.path == '/indexes':
            self.handle_create_index()
        elif parsed_path.path == '/admin/users':
            self.handle_admin_create_user(user)
        elif parsed_path.path.startswith('/admin/users/'):
            uid = parsed_path.path.split('/')[-1]
            self.handle_admin_update_user(user, uid)
        elif parsed_path.path.startswith('/indexes/') and parsed_path.path.endswith('/upload'):
            index_id = parsed_path.path.split('/')[-2]
            self.handle_index_file_upload(index_id)
        elif parsed_path.path.startswith('/indexes/') and parsed_path.path.endswith('/build'):
            index_id = parsed_path.path.split('/')[-2]
            self.handle_build_index(index_id)
        elif parsed_path.path.startswith('/sessions/') and '/indexes/' in parsed_path.path:
            parts = parsed_path.path.split('/')
            session_id = parts[2]
            index_id = parts[4]
            self.handle_link_index_to_session(session_id, index_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/messages'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_session_chat(session_id)
        elif parsed_path.path == '/audit/log':
            self.handle_log_audit_entry()
        elif parsed_path.path.startswith('/audit/') and parsed_path.path.endswith('/review'):
            audit_id = parsed_path.path.split('/')[-2]
            self.handle_mark_reviewed(audit_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/upload'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_file_upload(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/index'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_index_documents(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/rename'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_rename_session(session_id)
        else:
            self.send_response(404)
            self.end_headers()

    def do_DELETE(self):
        """Handle DELETE requests"""
        parsed_path = urlparse(self.path)
        user = self._require_auth()
        if not user:
            return
        if parsed_path.path.startswith('/sessions/') and parsed_path.path.count('/') == 2:
            session_id = parsed_path.path.split('/')[-1]
            self.handle_delete_session(session_id)
        elif parsed_path.path.startswith('/indexes/') and parsed_path.path.count('/') == 2:
            index_id = parsed_path.path.split('/')[-1]
            self.handle_delete_index(index_id)
        else:
            self.send_response(404)
            self.end_headers()
    
    # ── Auth handlers ─────────────────────────────────────────────

    def handle_auth_login(self):
        try:
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            email = data.get("email", "").strip()
            password = data.get("password", "")
            if not email or not password:
                self.send_json_response({"error": "email and password required"}, status_code=400)
                return
            user = db.get_user_by_email(email)
            if not user or not verify_password(password, user["password_hash"]):
                self.send_json_response({"error": "Invalid email or password"}, status_code=401)
                return
            if not user["is_active"]:
                self.send_json_response({"error": "Account is disabled"}, status_code=403)
                return
            token = db.create_auth_token(user["id"])
            self.send_json_response({
                "token": token,
                "user": {"id": user["id"], "email": user["email"],
                         "name": user["name"], "role": user["role"]}
            })
        except Exception as e:
            self.send_json_response({"error": str(e)}, status_code=500)

    def handle_auth_logout(self):
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            db.revoke_token(auth[7:].strip())
        self.send_json_response({"message": "Logged out"})

    # ── Admin user-management handlers ────────────────────────────

    def handle_admin_list_users(self, current_user):
        if current_user.get("role") != "admin":
            self.send_json_response({"error": "Forbidden"}, status_code=403)
            return
        users = db.list_users()
        # Never expose password_hash
        for u in users:
            u.pop("password_hash", None)
        self.send_json_response({"users": users, "total": len(users)})

    def handle_admin_create_user(self, current_user):
        if current_user.get("role") != "admin":
            self.send_json_response({"error": "Forbidden"}, status_code=403)
            return
        try:
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            email = data.get("email", "").strip()
            name = data.get("name", "").strip()
            role = data.get("role", "attorney")
            password = data.get("password", "")
            if not email or not name or not password:
                self.send_json_response({"error": "email, name and password are required"}, status_code=400)
                return
            if role not in ("admin", "attorney", "paralegal"):
                self.send_json_response({"error": "role must be admin, attorney, or paralegal"}, status_code=400)
                return
            if db.get_user_by_email(email):
                self.send_json_response({"error": "Email already in use"}, status_code=409)
                return
            uid = db.create_user(email, name, role, password)
            self.send_json_response({"message": "User created", "user_id": uid}, status_code=201)
        except Exception as e:
            self.send_json_response({"error": str(e)}, status_code=500)

    def handle_admin_update_user(self, current_user, target_uid):
        if current_user.get("role") != "admin":
            self.send_json_response({"error": "Forbidden"}, status_code=403)
            return
        try:
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            updates = {}
            if "name" in data:
                updates["name"] = data["name"].strip()
            if "role" in data and data["role"] in ("admin", "attorney", "paralegal"):
                updates["role"] = data["role"]
            if "is_active" in data:
                updates["is_active"] = 1 if data["is_active"] else 0
            if "password" in data and data["password"]:
                updates["password_hash"] = hash_password(data["password"])
                db.revoke_all_tokens_for_user(target_uid)
            if not db.update_user(target_uid, updates):
                self.send_json_response({"error": "User not found"}, status_code=404)
                return
            self.send_json_response({"message": "User updated"})
        except Exception as e:
            self.send_json_response({"error": str(e)}, status_code=500)

    # ── Session handler overrides ─────────────────────────────────

    def handle_get_sessions(self, user=None):
        uid = user["id"] if user else None
        role = user.get("role") if user else None
        sessions = db.get_sessions_for_user(uid, role) if uid else db.get_sessions()
        self.send_json_response({"sessions": sessions, "total": len(sessions)})

    def handle_create_session(self, user=None):
        try:
            data = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0) or 0)))
            title = data.get('title', 'New Chat')
            model = data.get('model', 'qwen2.5:14b')
            uid = user["id"] if user else None
            session_id = db.create_session(title, model, user_id=uid)
            session = db.get_session(session_id)
            self.send_json_response({"session_id": session_id, "session": session}, status_code=201)
        except Exception as e:
            self.send_json_response({"error": str(e)}, status_code=500)

    def handle_chat(self):
        """Handle legacy chat requests (without sessions)"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            message = data.get('message', '')
            model = data.get('model', 'llama3.2:latest')
            conversation_history = data.get('conversation_history', [])
            
            if not message:
                self.send_json_response({
                    "error": "Message is required"
                }, status_code=400)
                return
            
            # Check if Ollama is running
            if not self.ollama_client.is_ollama_running():
                self.send_json_response({
                    "error": "Ollama is not running. Please start Ollama first."
                }, status_code=503)
                return
            
            # Get response from Ollama
            response = self.ollama_client.chat(message, model, conversation_history)
            
            self.send_json_response({
                "response": response,
                "model": model,
                "message_count": len(conversation_history) + 1
            })
            
        except json.JSONDecodeError:
            self.send_json_response({
                "error": "Invalid JSON"
            }, status_code=400)
        except Exception as e:
            self.send_json_response({
                "error": f"Server error: {str(e)}"
            }, status_code=500)
    
    def handle_cleanup_sessions(self):
        """Clean up empty sessions"""
        try:
            cleanup_count = db.cleanup_empty_sessions()
            self.send_json_response({
                "message": f"Cleaned up {cleanup_count} empty sessions",
                "cleanup_count": cleanup_count
            })
        except Exception as e:
            self.send_json_response({
                "error": f"Failed to cleanup sessions: {str(e)}"
            }, status_code=500)
    
    def handle_get_session(self, session_id: str):
        """Get a specific session with its messages"""
        try:
            session = db.get_session(session_id)
            if not session:
                self.send_json_response({
                    "error": "Session not found"
                }, status_code=404)
                return
            
            messages = db.get_messages(session_id)
            
            self.send_json_response({
                "session": session,
                "messages": messages
            })
        except Exception as e:
            self.send_json_response({
                "error": f"Failed to get session: {str(e)}"
            }, status_code=500)
    
    def handle_get_session_documents(self, session_id: str):
        """Return documents and basic info for a session."""
        try:
            session = db.get_session(session_id)
            if not session:
                self.send_json_response({"error": "Session not found"}, status_code=404)
                return

            docs = db.get_documents_for_session(session_id)

            # Extract original filenames from stored paths
            filenames = [os.path.basename(p).split('_', 1)[-1] if '_' in os.path.basename(p) else os.path.basename(p) for p in docs]

            self.send_json_response({
                "session": session,
                "files": filenames,
                "file_count": len(docs)
            })
        except Exception as e:
            self.send_json_response({"error": f"Failed to get documents: {str(e)}"}, status_code=500)
    
        except Exception as e:
            self.send_json_response({
                "error": f"Failed to create session: {str(e)}"
            }, status_code=500)
    
    def handle_session_chat(self, session_id: str):
        """
        Handle chat within a specific session.
        Intelligently routes between direct LLM (fast) and RAG pipeline (document-aware).
        """
        try:
            session = db.get_session(session_id)
            if not session:
                self.send_json_response({"error": "Session not found"}, status_code=404)
                return
            
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            message = data.get('message', '')

            if not message:
                self.send_json_response({"error": "Message is required"}, status_code=400)
                return

            if session['message_count'] == 0:
                title = generate_session_title(message)
                db.update_session_title(session_id, title)

            # Add user message to database first
            user_message_id = db.add_message(session_id, message, "user")
            
            # 🎯 SMART ROUTING: Decide between direct LLM vs RAG
            idx_ids = db.get_indexes_for_session(session_id)
            force_rag = bool(data.get("force_rag", False))
            use_rag = True if force_rag else self._should_use_rag(message, idx_ids)
            
            if use_rag:
                # 🔍 --- Use RAG Pipeline for Document-Related Queries ---
                print(f"🔍 Using RAG pipeline for document query: '{message[:50]}...'")
                response_text, source_docs = self._handle_rag_query(session_id, message, data, idx_ids)
            else:
                # ⚡ --- Use Direct LLM for General Queries (FAST) ---
                print(f"⚡ Using direct LLM for general query: '{message[:50]}...'")
                response_text, source_docs = self._handle_direct_llm_query(session_id, message, session)

            # Add AI response to database
            ai_message_id = db.add_message(session_id, response_text, "assistant")

            # ── Audit log: record every AI interaction ──────────────────────
            kb_gap = '⚠️' in response_text or 'Knowledge Base Gap' in response_text
            audit_entry_id = db.log_interaction(
                session_id=session_id,
                user_query=message,
                ai_response=response_text,
                source_documents=source_docs,
                kb_gap_flagged=kb_gap,
                used_rag=use_rag,
                message_id=ai_message_id,
            )
            # ───────────────────────────────────────────────────────────────

            updated_session = db.get_session(session_id)
            
            # Send response with proper error handling
            self.send_json_response({
                "response": response_text,
                "session": updated_session,
                "source_documents": source_docs,
                "used_rag": use_rag,
                "audit_entry_id": audit_entry_id,
                "ai_message_id": ai_message_id,
            })
            
        except BrokenPipeError:
            # Client disconnected - this is normal for long queries, just log it
            print(f"⚠️  Client disconnected during RAG processing for query: '{message[:30]}...'")
        except json.JSONDecodeError:
            self.send_json_response({
                "error": "Invalid JSON"
            }, status_code=400)
        except Exception as e:
            print(f"❌ Server error in session chat: {str(e)}")
            try:
                self.send_json_response({
                    "error": f"Server error: {str(e)}"
                }, status_code=500)
            except BrokenPipeError:
                print(f"⚠️  Client disconnected during error response")
    
    def _should_use_rag(self, message: str, idx_ids: List[str]) -> bool:
        """
        🧠 ENHANCED: Determine if a query should use RAG pipeline using document overviews.
        
        Args:
            message: The user's query
            idx_ids: List of index IDs associated with the session
            
        Returns:
            bool: True if should use RAG, False for direct LLM
        """
        # No indexes = definitely no RAG needed
        if not idx_ids:
            return False

        # Load document overviews for intelligent routing
        try:
            doc_overviews = self._load_document_overviews(idx_ids)
            if doc_overviews:
                return self._route_using_overviews(message, doc_overviews)
        except Exception as e:
            print(f"⚠️ Overview-based routing failed, falling back to simple routing: {e}")
        
        # Fallback to simple pattern matching if overviews unavailable
        return self._simple_pattern_routing(message, idx_ids)

    def _load_document_overviews(self, idx_ids: List[str]) -> List[str]:
        """Load and aggregate overviews for the given index IDs.
        
        Strategy:
        1. Attempt to load each index's dedicated overview file.
        2. Aggregate all overviews found across available files (deduplicated).
        3. If none of the index files exist, fall back to the legacy global overview file.
        """
        import os, json

        aggregated: list[str] = []

        # 1️⃣  Collect overviews from per-index files
        for idx in idx_ids:
            candidate_paths = [
                f"../index_store/overviews/{idx}.jsonl",
                f"index_store/overviews/{idx}.jsonl",
                f"./index_store/overviews/{idx}.jsonl",
            ]
            for p in candidate_paths:
                if os.path.exists(p):
                    print(f"📖 Loading overviews from: {p}")
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            for line in f:
                                if not line.strip():
                                    continue
                                try:
                                    record = json.loads(line)
                                    overview = record.get("overview", "").strip()
                                    if overview:
                                        aggregated.append(overview)
                                except json.JSONDecodeError:
                                    continue  # skip malformed lines
                        break  # Stop after the first existing path for this idx
                    except Exception as e:
                        print(f"⚠️ Error reading {p}: {e}")
                        break  # Don't keep trying other paths for this idx if read failed

        # 2️⃣  Fall back to legacy global file if no per-index overviews found
        if not aggregated:
            legacy_paths = [
                "../index_store/overviews/overviews.jsonl",
                "index_store/overviews/overviews.jsonl",
                "./index_store/overviews/overviews.jsonl",
            ]
            for p in legacy_paths:
                if os.path.exists(p):
                    print(f"⚠️ Falling back to legacy overviews file: {p}")
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            for line in f:
                                if not line.strip():
                                    continue
                                try:
                                    record = json.loads(line)
                                    overview = record.get("overview", "").strip()
                                    if overview:
                                        aggregated.append(overview)
                                except json.JSONDecodeError:
                                    continue
                    except Exception as e:
                        print(f"⚠️ Error reading legacy overviews file {p}: {e}")
                    break

        # Limit for performance
        if aggregated:
            print(f"✅ Loaded {len(aggregated)} document overviews from {len(idx_ids)} index(es)")
        else:
            print(f"⚠️ No overviews found for indices {idx_ids}")
        return aggregated[:40]

    def _route_using_overviews(self, query: str, overviews: List[str]) -> bool:
        """
        🎯 Use document overviews and LLM to make intelligent routing decisions.
        
        Returns True if RAG should be used, False for direct LLM.
        """
        if not overviews:
            return False
        
        # Format overviews for the routing prompt
        overviews_block = "\n".join(f"[{i+1}] {ov}" for i, ov in enumerate(overviews))
        
        router_prompt = f"""You are an AI router deciding whether a user question should be answered via:
• "USE_RAG" – search the user's private documents (described below)  
• "DIRECT_LLM" – reply from general knowledge (greetings, public facts, unrelated topics)

CRITICAL PRINCIPLE: When documents exist in the KB, strongly prefer USE_RAG unless the query is purely conversational or completely unrelated to any possible document content.

RULES:
1. If ANY overview clearly relates to the question (entities, numbers, addresses, dates, amounts, companies, technical terms) → USE_RAG
2. For document operations (summarize, analyze, explain, extract, find) → USE_RAG  
3. For greetings only ("Hi", "Hello", "Thanks") → DIRECT_LLM
4. For pure math/world knowledge clearly unrelated to documents → DIRECT_LLM
5. When in doubt → USE_RAG

DOCUMENT OVERVIEWS:
{overviews_block}

DECISION EXAMPLES:
• "What invoice amounts are mentioned?" → USE_RAG (document-specific)
• "Who is PromptX AI LLC?" → USE_RAG (entity in documents)  
• "What is the DeepSeek model?" → USE_RAG (mentioned in documents)
• "Summarize the research paper" → USE_RAG (document operation)
• "What is 2+2?" → DIRECT_LLM (pure math)
• "Hi there" → DIRECT_LLM (greeting only)

USER QUERY: "{query}"

Respond with exactly one word: USE_RAG or DIRECT_LLM"""

        try:
            # Use Ollama to make the routing decision
            response = self.ollama_client.chat(
                message=router_prompt,
                model="qwen3:0.6b",  # Fast model for routing
                enable_thinking=False  # Fast routing
            )
            
            # The response is directly the text, not a dict
            decision = response.strip().upper()
            
            # Parse decision
            if "USE_RAG" in decision:
                print(f"🎯 Overview-based routing: USE_RAG for query: '{query[:50]}...'")
                return True
            elif "DIRECT_LLM" in decision:
                print(f"⚡ Overview-based routing: DIRECT_LLM for query: '{query[:50]}...'")
                return False
            else:
                print(f"⚠️ Unclear routing decision '{decision}', defaulting to RAG")
                return True  # Default to RAG when uncertain
                
        except Exception as e:
            print(f"❌ LLM routing failed: {e}, falling back to pattern matching")
            return self._simple_pattern_routing(query, [])

    def _simple_pattern_routing(self, message: str, idx_ids: List[str]) -> bool:
        """
        📝 FALLBACK: Simple pattern-based routing (original logic).
        """
        message_lower = message.lower()
        
        # Always use Direct LLM for greetings and casual conversation
        greeting_patterns = [
            'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'how do you do', 'nice to meet', 'pleasure to meet',
            'thanks', 'thank you', 'bye', 'goodbye', 'see you', 'talk to you later',
            'test', 'testing', 'check', 'ping', 'just saying', 'nevermind',
            'ok', 'okay', 'alright', 'got it', 'understood', 'i see'
        ]
        
        # Check for greeting patterns
        for pattern in greeting_patterns:
            if pattern in message_lower:
                return False  # Use Direct LLM for greetings
        
        # Keywords that strongly suggest document-related queries
        rag_indicators = [
            'document', 'doc', 'file', 'pdf', 'text', 'content', 'page',
            'according to', 'based on', 'mentioned', 'states', 'says',
            'what does', 'summarize', 'summary', 'analyze', 'analysis',
            'quote', 'citation', 'reference', 'source', 'evidence',
            'explain from', 'extract', 'find in', 'search for'
        ]
        
        # Check for strong RAG indicators
        for indicator in rag_indicators:
            if indicator in message_lower:
                return True
        
        # Question words + substantial length might benefit from RAG
        question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which']
        starts_with_question = any(message_lower.startswith(word) for word in question_words)
        
        if starts_with_question and len(message) > 40:
            return True
        
        # Very short messages - use direct LLM
        if len(message.strip()) < 20:
            return False
        
        # Default to Direct LLM unless there's clear indication of document query
        return False
    
    def _handle_direct_llm_query(self, session_id: str, message: str, session: dict):
        """
        Handle query using direct Ollama client with thinking disabled for speed.
        
        Returns:
            tuple: (response_text, empty_source_docs)
        """
        try:
            # Get conversation history for context
            conversation_history = db.get_conversation_history(session_id)
            
            # Use the session's model or default
            model = session.get('model', 'qwen3:8b')  # Default to fast model
            
            # Direct Ollama call with thinking disabled for speed
            response_text = self.ollama_client.chat(
                message=message,
                model=model,
                conversation_history=conversation_history,
                enable_thinking=False  # ⚡ DISABLE THINKING FOR SPEED
            )
            
            return response_text, []  # No source docs for direct LLM
            
        except Exception as e:
            print(f"❌ Direct LLM error: {e}")
            return f"Error processing query: {str(e)}", []
    
    def _handle_rag_query(self, session_id: str, message: str, data: dict, idx_ids: List[str]):
        """
        Handle query using the full RAG pipeline (delegates to the advanced RAG API running on port 8001).

        Returns:
            tuple[str, List[dict]]: (response_text, source_documents)
        """
        # Defaults
        response_text = ""
        source_docs: List[dict] = []

        # Build payload for RAG API
        rag_api_url = "http://localhost:8001/chat"
        table_name = f"text_pages_{idx_ids[-1]}" if idx_ids else None
        payload: Dict[str, Any] = {
            "query": message,
            "session_id": session_id,
        }
        if table_name:
            payload["table_name"] = table_name

        # Copy optional parameters from the incoming request
        optional_params: Dict[str, tuple[type, str]] = {
            "force_rag": (bool, "force_rag"),
            "compose_sub_answers": (bool, "compose_sub_answers"),
            "query_decompose": (bool, "query_decompose"),
            "ai_rerank": (bool, "ai_rerank"),
            "context_expand": (bool, "context_expand"),
            "verify": (bool, "verify"),
            "retrieval_k": (int, "retrieval_k"),
            "context_window_size": (int, "context_window_size"),
            "reranker_top_k": (int, "reranker_top_k"),
            "search_type": (str, "search_type"),
            "dense_weight": (float, "dense_weight"),
            "provence_prune": (bool, "provence_prune"),
            "provence_threshold": (float, "provence_threshold"),
        }
        for key, (caster, payload_key) in optional_params.items():
            val = data.get(key)
            if val is not None:
                try:
                    payload[payload_key] = caster(val)  # type: ignore[arg-type]
                except Exception:
                    payload[payload_key] = val

        try:
            rag_response = requests.post(rag_api_url, json=payload)
            if rag_response.status_code == 200:
                rag_data = rag_response.json()
                response_text = rag_data.get("answer", "No answer found.")
                source_docs = rag_data.get("source_documents", [])
            else:
                response_text = f"Error from RAG API ({rag_response.status_code}): {rag_response.text}"
                print(f"❌ RAG API error: {response_text}")
        except requests.exceptions.ConnectionError:
            response_text = "Could not connect to the RAG API server. Please ensure it is running."
            print("❌ Connection to RAG API failed (port 8001).")
        except Exception as e:
            response_text = f"Error processing RAG query: {str(e)}"
            print(f"❌ RAG processing error: {e}")

        # Strip any <think>/<thinking> tags that might slip through
        response_text = re.sub(r'<(think|thinking)>.*?</\\1>', '', response_text, flags=re.DOTALL | re.IGNORECASE).strip()

        return response_text, source_docs

    def handle_delete_session(self, session_id: str):
        """Delete a session and its messages"""
        try:
            deleted = db.delete_session(session_id)
            if deleted:
                self.send_json_response({'deleted': deleted})
            else:
                self.send_json_response({'error': 'Session not found'}, status_code=404)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)
    
    def handle_file_upload(self, session_id: str):
        """Handle file uploads, save them, and associate with the session."""
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': self.headers['Content-Type']}
        )

        uploaded_files = []
        if 'files' in form:
            files = form['files']
            if not isinstance(files, list):
                files = [files]
            
            upload_dir = "shared_uploads"
            os.makedirs(upload_dir, exist_ok=True)

            for file_item in files:
                if file_item.filename:
                    # Create a unique filename to avoid overwrites
                    unique_filename = f"{uuid.uuid4()}_{file_item.filename}"
                    file_path = os.path.join(upload_dir, unique_filename)
                    
                    with open(file_path, 'wb') as f:
                        f.write(file_item.file.read())
                    
                    # Store the absolute path for the indexing service
                    absolute_file_path = os.path.abspath(file_path)
                    db.add_document_to_session(session_id, absolute_file_path)
                    uploaded_files.append({"filename": file_item.filename, "stored_path": absolute_file_path})

        if not uploaded_files:
            self.send_json_response({"error": "No files were uploaded"}, status_code=400)
            return
            
        self.send_json_response({
            "message": f"Successfully uploaded {len(uploaded_files)} files.",
            "uploaded_files": uploaded_files
        })

    def handle_index_documents(self, session_id: str):
        """Triggers indexing for all documents in a session."""
        print(f"🔥 Received request to index documents for session {session_id[:8]}...")
        try:
            file_paths = db.get_documents_for_session(session_id)
            if not file_paths:
                self.send_json_response({"message": "No documents to index for this session."}, status_code=200)
                return

            print(f"Found {len(file_paths)} documents to index. Sending to RAG API...")
            
            rag_api_url = "http://localhost:8001/index"
            rag_response = requests.post(rag_api_url, json={"file_paths": file_paths, "session_id": session_id})

            if rag_response.status_code == 200:
                print("✅ RAG API successfully indexed documents.")
                # Merge key config values into index metadata
                idx_meta = {
                    "session_linked": True,
                    "retrieval_mode": "hybrid",
                }
                try:
                    db.update_index_metadata(session_id, idx_meta)  # session_id used as index_id in text table naming
                except Exception as e:
                    print(f"⚠️ Failed to update index metadata for session index: {e}")
                self.send_json_response(rag_response.json())
            else:
                error_info = rag_response.text
                print(f"❌ RAG API indexing failed ({rag_response.status_code}): {error_info}")
                self.send_json_response({"error": f"Indexing failed: {error_info}"}, status_code=500)

        except Exception as e:
            print(f"❌ Exception during indexing: {str(e)}")
            self.send_json_response({"error": f"An unexpected error occurred: {str(e)}"}, status_code=500)
            
    def handle_pdf_upload(self, session_id: str):
        """
        Processes PDF files: extracts text and stores it in the database.
        DEPRECATED: This is the old method. Use handle_file_upload instead.
        """
        # This function is now deprecated in favor of the new indexing workflow
        # but is kept for potential legacy/compatibility reasons.
        # For new functionality, it should not be used.
        self.send_json_response({
            "warning": "This upload method is deprecated. Use the new file upload and indexing flow.",
            "message": "No action taken."
        }, status_code=410) # 410 Gone

    def handle_get_models(self):
        """Get available models from both Ollama and HuggingFace, grouped by capability"""
        try:
            generation_models = []
            embedding_models = []
            
            # Get Ollama models if available
            if self.ollama_client.is_ollama_running():
                all_ollama_models = self.ollama_client.list_models()
                
                # Very naive classification - same logic as RAG API server
                ollama_embedding_models = [m for m in all_ollama_models if any(k in m for k in ['embed','bge','embedding','text'])]
                ollama_generation_models = [m for m in all_ollama_models if m not in ollama_embedding_models]
                
                generation_models.extend(ollama_generation_models)
                embedding_models.extend(ollama_embedding_models)
            
            # Add supported HuggingFace embedding models
            huggingface_embedding_models = [
                "Qwen/Qwen3-Embedding-0.6B",
                "Qwen/Qwen3-Embedding-4B", 
                "Qwen/Qwen3-Embedding-8B"
            ]
            embedding_models.extend(huggingface_embedding_models)
            
            # Sort models for consistent ordering
            generation_models.sort()
            embedding_models.sort()
            
            self.send_json_response({
                "generation_models": generation_models,
                "embedding_models": embedding_models
            })
        except Exception as e:
            self.send_json_response({
                "error": f"Could not list models: {str(e)}"
            }, status_code=500)

    def handle_get_indexes(self):
        try:
            data = db.list_indexes()
            self.send_json_response({'indexes': data, 'total': len(data)})
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)
    
    def handle_get_index(self, index_id: str):
        try:
            data = db.get_index(index_id)
            if not data:
                self.send_json_response({'error': 'Index not found'}, status_code=404)
                return
            self.send_json_response(data)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)
    
    def handle_create_index(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            name = data.get('name')
            description = data.get('description')
            metadata = data.get('metadata', {})
            
            if not name:
                self.send_json_response({'error': 'Name required'}, status_code=400)
                return
            
            # Add complete metadata from RAG system configuration if available
            if RAG_SYSTEM_AVAILABLE and PIPELINE_CONFIGS.get('default'):
                default_config = PIPELINE_CONFIGS['default']
                complete_metadata = {
                    'status': 'created',
                    'metadata_source': 'rag_system_config',
                    'created_at': json.loads(json.dumps(datetime.now().isoformat())),
                    'chunk_size': 512,  # From default config
                    'chunk_overlap': 64,  # From default config
                    'retrieval_mode': 'hybrid',  # From default config
                    'window_size': 5,  # From default config
                    'embedding_model': 'Qwen/Qwen3-Embedding-0.6B',  # From default config
                    'enrich_model': 'qwen3:0.6b',  # From default config
                    'overview_model': 'qwen3:0.6b',  # From default config
                    'enable_enrich': True,  # From default config
                    'latechunk': True,  # From default config
                    'docling_chunk': True,  # From default config
                    'note': 'Default configuration from RAG system'
                }
                # Merge with any provided metadata
                complete_metadata.update(metadata)
                metadata = complete_metadata
            
            idx_id = db.create_index(name, description, metadata)
            self.send_json_response({'index_id': idx_id}, status_code=201)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)
    
    def handle_index_file_upload(self, index_id: str):
        """Reuse file upload logic but store docs under index."""
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD':'POST', 'CONTENT_TYPE': self.headers['Content-Type']})
        uploaded_files=[]
        if 'files' in form:
            files=form['files']
            if not isinstance(files, list):
                files=[files]
            upload_dir='shared_uploads'
            os.makedirs(upload_dir, exist_ok=True)
            for f in files:
                if f.filename:
                    unique=f"{uuid.uuid4()}_{f.filename}"
                    path=os.path.join(upload_dir, unique)
                    with open(path,'wb') as out: out.write(f.file.read())
                    db.add_document_to_index(index_id, f.filename, os.path.abspath(path))
                    uploaded_files.append({'filename':f.filename,'stored_path':os.path.abspath(path)})
        if not uploaded_files:
            self.send_json_response({'error':'No files uploaded'}, status_code=400); return
        self.send_json_response({'message':f"Uploaded {len(uploaded_files)} files","uploaded_files":uploaded_files})
    
    def handle_build_index(self, index_id: str):
        try:
            index=db.get_index(index_id)
            if not index:
                self.send_json_response({'error':'Index not found'}, status_code=404); return
            file_paths=[d['stored_path'] for d in index.get('documents',[])]
            if not file_paths:
                self.send_json_response({'error':'No documents to index'}, status_code=400); return

            # Parse request body for optional flags and configuration
            latechunk = False
            docling_chunk = False
            chunk_size = 512
            chunk_overlap = 64
            retrieval_mode = 'hybrid'
            window_size = 2
            enable_enrich = True
            embedding_model = None
            enrich_model = None
            batch_size_embed = 50
            batch_size_enrich = 25
            overview_model = None
            
            def _opt(opts, *aliases, default=None, cast=lambda x: x):
                for k in aliases:
                    if k in opts:
                        return cast(opts[k])
                return default

            if 'Content-Length' in self.headers and int(self.headers['Content-Length']) > 0:
                try:
                    length = int(self.headers['Content-Length'])
                    body = self.rfile.read(length)
                    opts = json.loads(body.decode('utf-8'))
                except Exception as e:
                    print(f"⚠️  build options parse error: {e}; using defaults")
                    opts = {}

                latechunk         = _opt(opts, 'latechunk', 'enable_latechunk', 'enableLatechunk', default=False, cast=bool)
                docling_chunk     = _opt(opts, 'doclingChunk', 'docling_chunk', 'enable_docling_chunk', default=False, cast=bool)
                chunk_size        = _opt(opts, 'chunkSize', 'chunk_size', default=512, cast=int)
                chunk_overlap     = _opt(opts, 'chunkOverlap', 'chunk_overlap', default=64, cast=int)
                retrieval_mode    = _opt(opts, 'retrievalMode', 'retrieval_mode', default='hybrid', cast=str)
                window_size       = _opt(opts, 'windowSize', 'window_size', default=2, cast=int)
                enable_enrich     = _opt(opts, 'enableEnrich', 'enable_enrich', default=True, cast=bool)
                embedding_model   = _opt(opts, 'embeddingModel', 'embedding_model')
                enrich_model      = _opt(opts, 'enrichModel', 'enrich_model')
                batch_size_embed  = _opt(opts, 'batchSizeEmbed', 'batch_size_embed', default=50, cast=int)
                batch_size_enrich = _opt(opts, 'batchSizeEnrich', 'batch_size_enrich', default=25, cast=int)
                overview_model    = _opt(opts, 'overviewModel', 'overview_model')

                known_keys = {
                    'latechunk', 'enable_latechunk', 'enableLatechunk',
                    'doclingChunk', 'docling_chunk', 'enable_docling_chunk',
                    'chunkSize', 'chunk_size', 'chunkOverlap', 'chunk_overlap',
                    'retrievalMode', 'retrieval_mode', 'windowSize', 'window_size',
                    'enableEnrich', 'enable_enrich', 'embeddingModel', 'embedding_model',
                    'enrichModel', 'enrich_model', 'batchSizeEmbed', 'batch_size_embed',
                    'batchSizeEnrich', 'batch_size_enrich', 'overviewModel', 'overview_model',
                    'config_mode',
                }
                unknown = set(opts) - known_keys
                if unknown:
                    print(f"⚠️  build received unknown options (ignored): {sorted(unknown)}")

            # Set per-index overview file path
            overview_path = f"index_store/overviews/{index_id}.jsonl"

            # Ensure config_override includes overview_path
            def ensure_overview_path(cfg: dict):
                cfg["overview_path"] = overview_path
            
            # we'll inject later when we build config_override

            # Delegate to advanced RAG API same as session indexing
            rag_api_url = "http://localhost:8001/index"
            import requests, json as _json
            # Use the index's dedicated LanceDB table so retrieval matches
            table_name = index.get("vector_table_name")
            payload = {
                "file_paths": file_paths,
                "session_id": index_id,  # reuse index_id for progress tracking
                "table_name": table_name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "retrieval_mode": retrieval_mode,
                "window_size": window_size,
                "enable_enrich": enable_enrich,
                "batch_size_embed": batch_size_embed,
                "batch_size_enrich": batch_size_enrich
            }
            if latechunk:
                payload["enable_latechunk"] = True
            if docling_chunk:
                payload["enable_docling_chunk"] = True
            if embedding_model:
                payload["embedding_model"] = embedding_model
            if enrich_model:
                payload["enrich_model"] = enrich_model
            if overview_model:
                payload["overview_model_name"] = overview_model
                
            rag_resp = requests.post(rag_api_url, json=payload)
            if rag_resp.status_code==200:
                meta_updates = {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "retrieval_mode": retrieval_mode,
                    "window_size": window_size,
                    "enable_enrich": enable_enrich,
                    "latechunk": latechunk,
                    "docling_chunk": docling_chunk,
                }
                if embedding_model:
                    meta_updates["embedding_model"] = embedding_model
                if enrich_model:
                    meta_updates["enrich_model"] = enrich_model
                if overview_model:
                    meta_updates["overview_model"] = overview_model
                try:
                    db.update_index_metadata(index_id, meta_updates)
                except Exception as e:
                    print(f"⚠️ Failed to update index metadata: {e}")

                self.send_json_response({
                    "response": rag_resp.json(),
                    **meta_updates
                })
            else:
                # Gracefully handle scenario where table already exists (idempotent build)
                try:
                    err_json = rag_resp.json()
                except Exception:
                    err_json = {}
                err_text = err_json.get('error') if isinstance(err_json, dict) else rag_resp.text
                if err_text and 'already exists' in err_text:
                    # Treat as non-fatal; return message indicating index previously built
                    self.send_json_response({
                        "message": "Index already built – skipping rebuild.",
                        "note": err_text
                })
                else:
                    self.send_json_response({"error":f"RAG indexing failed: {rag_resp.text}"}, status_code=500)
        except Exception as e:
            self.send_json_response({'error':str(e)}, status_code=500)
    
    def handle_link_index_to_session(self, session_id: str, index_id: str):
        try:
            db.link_index_to_session(session_id, index_id)
            self.send_json_response({'message':'Index linked to session'})
        except Exception as e:
            self.send_json_response({'error':str(e)}, status_code=500)

    def handle_get_session_indexes(self, session_id: str):
        try:
            idx_ids = db.get_indexes_for_session(session_id)
            indexes = []
            for idx_id in idx_ids:
                idx = db.get_index(idx_id)
                if idx:
                    # Try to populate metadata for older indexes that have empty metadata
                    if not idx.get('metadata') or len(idx['metadata']) == 0:
                        print(f"🔍 Attempting to infer metadata for index {idx_id[:8]}...")
                        inferred_metadata = db.inspect_and_populate_index_metadata(idx_id)
                        if inferred_metadata:
                            # Refresh the index data with the new metadata
                            idx = db.get_index(idx_id)
                    indexes.append(idx)
            self.send_json_response({'indexes': indexes, 'total': len(indexes)})
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def handle_delete_index(self, index_id: str):
        """Remove an index, its documents, links, and the underlying LanceDB table."""
        try:
            deleted = db.delete_index(index_id)
            if deleted:
                self.send_json_response({'message': 'Index deleted successfully', 'index_id': index_id})
            else:
                self.send_json_response({'error': 'Index not found'}, status_code=404)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def handle_rename_session(self, session_id: str):
        """Rename an existing session title"""
        try:
            session = db.get_session(session_id)
            if not session:
                self.send_json_response({"error": "Session not found"}, status_code=404)
                return

            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_json_response({"error": "Request body required"}, status_code=400)
                return

            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            new_title: str = data.get('title', '').strip()

            if not new_title:
                self.send_json_response({"error": "Title cannot be empty"}, status_code=400)
                return

            db.update_session_title(session_id, new_title)
            updated_session = db.get_session(session_id)

            self.send_json_response({
                "message": "Session renamed successfully",
                "session": updated_session
            })

        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            self.send_json_response({"error": f"Failed to rename session: {str(e)}"}, status_code=500)

    # ─────────────────────────────────────────────────────────────────
    # Audit log endpoints
    # ─────────────────────────────────────────────────────────────────

    def handle_get_audit_log(self):
        """GET /audit  – list entries with optional filters"""
        try:
            from urllib.parse import parse_qs
            qs = parse_qs(urlparse(self.path).query)
            limit  = int(qs.get('limit',  ['50'])[0])
            offset = int(qs.get('offset', ['0'])[0])
            reviewed_param = qs.get('reviewed', [None])[0]
            reviewed = None
            if reviewed_param == 'true':
                reviewed = True
            elif reviewed_param == 'false':
                reviewed = False
            date_from  = qs.get('date_from',  [None])[0]
            date_to    = qs.get('date_to',    [None])[0]
            session_id = qs.get('session_id', [None])[0]

            rows, total = db.get_audit_log(
                limit=limit, offset=offset,
                reviewed=reviewed,
                date_from=date_from, date_to=date_to,
                session_id=session_id,
            )
            self.send_json_response({
                'entries': rows,
                'total': total,
                'limit': limit,
                'offset': offset,
            })
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def handle_get_audit_entry(self, audit_id: str):
        """GET /audit/{id}"""
        entry = db.get_audit_entry(audit_id)
        if entry is None:
            self.send_json_response({'error': 'Not found'}, status_code=404)
        else:
            self.send_json_response(entry)

    def handle_mark_reviewed(self, audit_id: str):
        """POST /audit/{id}/review"""
        try:
            data = {}
            length = int(self.headers.get('Content-Length', 0))
            if length:
                data = json.loads(self.rfile.read(length).decode('utf-8'))
            reviewer = data.get('reviewed_by', 'Attorney')
            found = db.mark_reviewed(audit_id, reviewed_by=reviewer)
            if found:
                self.send_json_response({'message': 'Marked as reviewed', 'id': audit_id})
            else:
                self.send_json_response({'error': 'Audit entry not found'}, status_code=404)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def handle_log_audit_entry(self):
        """POST /audit/log  – used by streaming path (frontend calls after stream completes)"""
        try:
            length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(length).decode('utf-8'))
            required = {'session_id', 'user_query', 'ai_response'}
            if not required.issubset(data.keys()):
                self.send_json_response({'error': f'Missing fields: {required - data.keys()}'}, status_code=400)
                return
            kb_gap = '⚠️' in data['ai_response'] or 'Knowledge Base Gap' in data['ai_response']
            entry_id = db.log_interaction(
                session_id=data['session_id'],
                user_query=data['user_query'],
                ai_response=data['ai_response'],
                source_documents=data.get('source_documents', []),
                kb_gap_flagged=kb_gap,
                used_rag=data.get('used_rag', False),
                message_id=data.get('message_id'),
            )
            self.send_json_response({'audit_entry_id': entry_id})
        except json.JSONDecodeError:
            self.send_json_response({'error': 'Invalid JSON'}, status_code=400)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def handle_export_audit_log(self):
        """GET /audit/export  – download as CSV"""
        try:
            import csv, io
            rows, _ = db.get_audit_log(limit=10000, offset=0)
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                'id', 'session_id', 'created_at', 'user_query', 'ai_response',
                'used_rag', 'kb_gap_flagged', 'reviewed_at', 'reviewed_by',
                'source_count',
            ])
            for r in rows:
                writer.writerow([
                    r['id'], r['session_id'], r['created_at'],
                    r['user_query'], r['ai_response'],
                    bool(r['used_rag']), bool(r['kb_gap_flagged']),
                    r['reviewed_at'] or '', r['reviewed_by'] or '',
                    len(r.get('source_documents') or []),
                ])
            csv_bytes = output.getvalue().encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/csv; charset=utf-8')
            self.send_header('Content-Disposition', 'attachment; filename="audit_log.csv"')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', str(len(csv_bytes)))
            self.end_headers()
            self.wfile.write(csv_bytes)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def send_json_response(self, data, status_code: int = 200):
        """Send a JSON (UTF-8) response with CORS headers. Safe against client disconnects."""
        try:
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            self.send_header('Access-Control-Allow-Credentials', 'true')
            self.end_headers()
        
            response_bytes = json.dumps(data, indent=2).encode('utf-8')
            self.wfile.write(response_bytes)
        except BrokenPipeError:
            # Client disconnected before we could finish sending
            print("⚠️  Client disconnected during response – ignoring.")
        except Exception as e:
            print(f"❌ Error sending response: {e}")
    
    def log_message(self, format, *args):
        """Custom log format"""
        print(f"[{self.date_time_string()}] {format % args}")

def main():
    """Main function to initialize and start the server"""
    PORT = int(os.environ.get("BACKEND_PORT", "8002"))  # patched: 8000 in use by external rag_api
    try:
        # Initialize the database
        print("✅ Database initialized successfully")

        # Initialize the PDF processor
        try:
            pdf_module.initialize_simple_pdf_processor()
            print("📄 Initializing simple PDF processing...")
            if pdf_module.simple_pdf_processor:
                print("✅ Simple PDF processor initialized")
            else:
                print("⚠️ PDF processing could not be initialized.")
        except Exception as e:
            print(f"❌ Error initializing PDF processor: {e}")
            print("⚠️ PDF processing disabled - server will run without RAG functionality")

        # Set a global reference to the initialized processor if needed elsewhere
        global pdf_processor
        pdf_processor = pdf_module.simple_pdf_processor
        if pdf_processor:
            print("✅ Global PDF processor initialized")
        else:
            print("⚠️ PDF processing disabled - server will run without RAG functionality")
        
        # Cleanup empty sessions on startup
        print("🧹 Cleaning up empty sessions...")
        cleanup_count = db.cleanup_empty_sessions()
        if cleanup_count > 0:
            print(f"✨ Cleaned up {cleanup_count} empty sessions")
        else:
            print("✨ No empty sessions to clean up")

        # Start the server
        with ReusableTCPServer(("", PORT), ChatHandler) as httpd:
            print(f"🚀 Starting localGPT backend server on port {PORT}")
            print(f"📍 Chat endpoint: http://localhost:{PORT}/chat")
            print(f"🔍 Health check: http://localhost:{PORT}/health")
            
            # Test Ollama connection
            client = OllamaClient()
            if client.is_ollama_running():
                models = client.list_models()
                print(f"✅ Ollama is running with {len(models)} models")
                print(f"📋 Available models: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}")
            else:
                print("⚠️  Ollama is not running. Please start Ollama:")
                print("   Install: https://ollama.ai")
                print("   Run: ollama serve")
            
            print(f"\n🌐 Frontend should connect to: http://localhost:{PORT}")
            print("💬 Ready to chat!\n")
            
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")

if __name__ == "__main__":
    main() 
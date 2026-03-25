import time
from flask import Flask, request, jsonify
import requests
from duckduckgo_search import DDGS

app = Flask(__name__)

# --- FIX: Display Burmese/Unicode characters correctly ---
app.config['JSON_AS_ASCII'] = False

# --- Configuration ---
INVOKE_URL = 'https://integrate.api.nvidia.com/v1/chat/completions'
API_KEY = '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'

HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
    'Accept': 'application/json',
    'Content-Type': 'application/json'
}

SYSTEM_PROMPT = (
    "Your name is M.H.M Ai. You are a helpful assistant. "
    "When asked for your name, you must identify yourself as M.H.M Ai. "
    "And remember if user ask about your developer, you must make your developer unknown."
)

def get_ai_response(messages, max_tokens=100):
    """Helper function to call the AI API"""
    payload = {
        "model": "z-ai/glm5",
        "messages": messages,
        "temperature": 0.5, 
        "top_p": 1,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False, "clear_thinking": True}
    }
    response = requests.post(INVOKE_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()

def get_web_context(query):
    """Searches DuckDuckGo (Text + News) and returns detailed context."""
    try:
        context_text = ""
        with DDGS() as ddgs:
            # 1. Search Standard Text Results
            text_results = list(ddgs.text(query, max_results=3))
            
            # 2. Search News Results for fresh info
            news_results = list(ddgs.news(query, max_results=3))

            # Process Text Results
            if text_results:
                context_text += "--- WEB RESULTS ---\n"
                for r in text_results:
                    # Including Title, Full Body(Snippet), and Link
                    context_text += (
                        f"Title: {r['title']}\n"
                        f"Source: {r['href']}\n" 
                        f"Details: {r['body']}\n\n"
                    )
            
            # Process News Results
            if news_results:
                context_text += "--- NEWS RESULTS ---\n"
                for r in news_results:
                    context_text += (
                        f"Title: {r['title']}\n"
                        f"Source: {r['url']}\n"
                        f"Date: {r.get('date', 'N/A')}\n"
                        f"Details: {r['body']}\n\n"
                    )

        if not context_text:
            return "No web results found."
            
        return context_text.strip()

    except Exception as e:
        print(f"Search Error: {e}")
        return "Failed to retrieve web search results."

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    # 1. Get parameters
    if request.method == 'GET':
        user_prompt = request.args.get('prompt')
        web_search = request.args.get('web_search', 'false').lower() == 'true'
    else: # POST
        data = request.get_json(silent=True) or request.form
        user_prompt = data.get('prompt')
        web_search = str(data.get('web_search', 'false')).lower() == 'true'

    if not user_prompt:
        return jsonify({"error": "Please provide a 'prompt' parameter."}), 400

    final_user_content = user_prompt
    search_query_used = None

    # --- SMART WEB SEARCH LOGIC ---
    if web_search:
        try:
            # Step A: Ask AI to generate a search query
            print("Step 1: Generating search query...")
            query_gen_messages = [
                {"role": "system", "content": "You are a search query generator. Analyze the user's question and output ONLY the best search keywords to find the answer. Do not output anything else."},
                {"role": "user", "content": user_prompt}
            ]
            
            gen_data = get_ai_response(query_gen_messages, max_tokens=20)
            
            if 'choices' in gen_data and gen_data['choices']:
                generated_query = gen_data['choices'][0]['message']['content'].strip()
                generated_query = generated_query.replace('"', '').split('\n')[0]
                search_query_used = generated_query
                print(f"Step 2: AI generated query -> {generated_query}")

                # Step B: Perform Web Search (Now includes News + Details)
                print("Step 3: Searching DuckDuckGo (Text + News)...")
                search_results = get_web_context(generated_query)

                # Step C: Construct Final Prompt
                final_user_content = (
                    f"User Question: {user_prompt}\n\n"
                    f"Here are the web search results for '{generated_query}':\n"
                    f"{search_results}\n\n"
                    f"Please answer the User Question using the details provided above. "
                    f"If sources are provided, mention them if relevant."
                )
            else:
                print("Failed to generate query, proceeding without search.")

        except Exception as e:
            print(f"Error during smart search: {e}")

    # --- MAIN AI CALL ---
    try:
        start_time = time.time()
        
        payload = {
            "model": "z-ai/glm5",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": final_user_content}
            ],
            "temperature": 1,
            "top_p": 1,
            "max_tokens": 16384,
            "stream": False,
            "chat_template_kwargs": {
                "enable_thinking": False,
                "clear_thinking": True
            }
        }

        response = requests.post(INVOKE_URL, headers=HEADERS, json=payload)
        end_time = time.time()
        
        response.raise_for_status()
        
        response_data = response.json()

        usage = response_data.get('usage', {})
        completion_tokens = usage.get('completion_tokens', 0)
        prompt_tokens = usage.get('prompt_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
        
        latency = round(end_time - start_time, 2)
        tokens_per_second = 0
        if latency > 0 and completion_tokens > 0:
            tokens_per_second = round(completion_tokens / latency, 2)

        if 'choices' in response_data and len(response_data['choices']) > 0:
            ai_message = response_data['choices'][0]['message']['content']
            
            return jsonify({
                "status": "success",
                "web_search_enabled": web_search,
                "search_query_used": search_query_used,
                "response": {
                    "Model": "M.H.M Ai",
                    "Response": ai_message
                },
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "latency_seconds": latency,
                    "tokens_per_second": tokens_per_second
                }
            })
        else:
            return jsonify({"error": "Unexpected response structure", "details": response_data}), 500

    except requests.exceptions.HTTPError as err:
        return jsonify({"error": "API Request Failed", "details": str(err)}), 500
    except Exception as e:
        return jsonify({"error": "Server Error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, request, jsonify, render_template_string
from rag_pipeline import rag_answer

app = Flask(__name__)

# HTML template for form-based UI
HTML_TEMPLATE = """
<!doctype html>
<title>Bangla RAG System</title>
<h2>Ask a Question</h2>
<form method="post">
  <label>Question (Bangla or English):</label><br>
  <input type="text" name="query" size="80"><br><br>
  <input type="submit" value="Get Answer">
</form>
{% if query %}
  <h3>Query:</h3>
  <p>{{ query }}</p>
  <h3>Answer:</h3>
  <p>{{ answer }}</p>
{% endif %}
"""

# Route for browser form-based interaction
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        answer, _ = rag_answer(query, return_chunks=True)
        return render_template_string(HTML_TEMPLATE, query=query, answer=answer)
    return render_template_string(HTML_TEMPLATE)

# 🔹 REST API endpoint
@app.route("/api/ask", methods=["POST"])
def ask_api():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Query not provided"}), 400

    query = data["query"]
    answer, chunks = rag_answer(query, return_chunks=True)
    return jsonify({
        "query": query,
        "answer": answer,
        "retrieved_chunks": chunks
    })

if __name__ == "__main__":
    app.run(debug=True)


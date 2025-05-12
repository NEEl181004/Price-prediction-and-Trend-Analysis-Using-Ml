from flask import Flask, request, jsonify, send_from_directory
from Pm import compute_cosine_similarity, fetch_product_data
from flask_cors import CORS
from Graph import generate_price_chart  # Ensure this is implemented correctly
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
app = Flask(__name__)
CORS(app)

# Ensure charts folder exists
os.makedirs('charts', exist_ok=True)

# === Pattern Matching Endpoint ===
@app.route('/api/pattern-matching', methods=['GET'])
def pattern_matching():
    query = request.args.get('query', '')

    # Fetch product data
    product_data = fetch_product_data()
    product_titles = product_data['product_name'].tolist()

    # Perform pattern matching
    distances, indices = compute_cosine_similarity([query] + product_titles)

    # Prepare result
    top_match_index = indices[0][1]  # index 0 is the query itself
    top_match = product_titles[top_match_index]

    return jsonify({'query': query, 'top_match': top_match})



@app.route('/api/generate_chart', methods=['GET'])
def generate_chart():
    product_name = request.args.get('product_name')
    platform = request.args.get('platform')

    if not product_name or not platform:
        return jsonify({"error": "Missing product_name or platform"}), 400

    filename_safe = product_name.replace(' ', '_') + f"_{platform.lower()}_chart.html"
    chart_path = os.path.join('charts', filename_safe)
    chart_filename = filename_safe  # this is what frontend uses
    

    success, prediction, category, summary, chart_data = generate_price_chart(product_name, platform)

    if not success:
        return jsonify({"error": "No data found for this product and platform."}), 404

    # Ensure all values are JSON serializable
    clean_predictions = []
    for item in prediction:
        clean_predictions.append({
            "month": item["month"],
            "model_name": item["model_name"],
            "predicted_price": float(item["predicted_price"]),
            "timestamp": item.get("timestamp", "")  # Optional: add if needed for rendering summary
        })

    return jsonify({
        "success": True,
        "prediction": clean_predictions,
        "category": category,
        "summary": summary,  # make sure summary is a dict already
        "chart_filename": chart_filename  # this lets the frontend load the chart
    })



    return True, prediction, category, summary, chart_data


# === Serve Chart Files ===
@app.route('/charts/<path:filename>')
def serve_chart(filename):
    return send_from_directory('static/charts', filename)

# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)
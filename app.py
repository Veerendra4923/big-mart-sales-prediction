from flask import Flask, request, render_template
import numpy as np
import pickle

# Flask app initialization
app = Flask(__name__)

# Load all models
linear_model = pickle.load(open('linear_model.pkl', 'rb'))
poly_model = pickle.load(open('poly_model.pkl', 'rb'))
ridge_model = pickle.load(open('ridge_model.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
gb_model = pickle.load(open('gb_model.pkl', 'rb'))
cat_model = pickle.load(open('cat_model.pkl', 'rb'))

# For polynomial features transformation
poly_features = pickle.load(open('poly_features.pkl', 'rb'))
# For scaling (needed for linear models)
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        Item_Identifier = float(request.form["Item_Identifier"])
        Item_weight = float(request.form["Item_weight"])
        Item_Fat_Content = float(request.form["Item_Fat_Content"])
        Item_visibility = float(request.form["Item_visibility"])
        Item_Type = float(request.form["Item_Type"])
        Item_MPR = float(request.form["Item_MPR"])
        Outlet_identifier = float(request.form["Outlet_identifier"])
        Outlet_established_year = float(request.form["Outlet_established_year"])
        Outlet_size = float(request.form["Outlet_size"])
        Outlet_location_type = float(request.form["Outlet_location_type"])
        Outlet_type = float(request.form["Outlet_type"])
        
        # Get selected model
        selected_model = request.form.get("model_selection", "linear")

        # Create feature array
        features = np.array([[Item_Identifier, Item_weight, Item_Fat_Content, 
                              Item_visibility, Item_Type, Item_MPR, 
                              Outlet_identifier, Outlet_established_year, 
                              Outlet_size, Outlet_location_type, Outlet_type]])

        # Make prediction based on selected model
        warning = None
        if selected_model == "linear":
            # Scale features for linear model
            features_scaled = scaler.transform(features)
            raw_prediction = linear_model.predict(features_scaled)[0]
            # Ensure non-negative prediction
            prediction = max(0, raw_prediction)
            if raw_prediction < 0:
                warning = f"Original prediction was negative (₹{raw_prediction:.2f}). Adjusted to non-negative value."
        elif selected_model == "polynomial":
            # Scale and transform to polynomial features
            features_scaled = scaler.transform(features)
            features_poly = poly_features.transform(features_scaled)
            raw_prediction = poly_model.predict(features_poly)[0]
            # Ensure non-negative prediction
            prediction = max(0, raw_prediction)
            if raw_prediction < 0:
                warning = f"Original prediction was negative (₹{raw_prediction:.2f}). Adjusted to non-negative value."
        elif selected_model == "ridge":
            # Scale features for ridge model
            features_scaled = scaler.transform(features)
            raw_prediction = ridge_model.predict(features_scaled)[0]
            # Ensure non-negative prediction
            prediction = max(0, raw_prediction)
            if raw_prediction < 0:
                warning = f"Original prediction was negative (₹{raw_prediction:.2f}). Adjusted to non-negative value."
        elif selected_model == "random_forest":
            prediction = rf_model.predict(features)[0]
        elif selected_model == "gradient_boosting":
            prediction = gb_model.predict(features)[0]
        elif selected_model == "catboost":
            prediction = cat_model.predict(features)[0]
        else:
            # Default to random forest if no valid model selected
            prediction = rf_model.predict(features)[0]

        # Check for unusually low predictions
        if 0 <= prediction < 1000:  # Adjust this threshold based on your data
            if not warning:
                warning = "Prediction is unusually low. Consider using a different model or reviewing input data."

        # Generate sales improvement suggestions based on input parameters
        improvement_suggestions = generate_improvement_suggestions(
            Item_visibility, Item_MPR, Item_Type, Item_Fat_Content, 
            Outlet_size, Outlet_location_type, Outlet_type
        )
        
        # Simulate potential improvement percentages for different strategies
        visibility_increase = simulate_improved_sales(features, "visibility", selected_model)
        price_adjustment = simulate_improved_sales(features, "price", selected_model)
        outlet_optimization = simulate_improved_sales(features, "outlet", selected_model)
        
        # Format prediction with currency formatting
        formatted_prediction = f"₹{prediction:.2f}"
        
        return render_template('index.html', 
                               prediction=formatted_prediction, 
                               selected_model=selected_model,
                               warning=warning,
                               improvement_suggestions=improvement_suggestions,
                               visibility_increase=visibility_increase,
                               price_adjustment=price_adjustment,
                               outlet_optimization=outlet_optimization)

def generate_improvement_suggestions(visibility, mpr, item_type, fat_content, outlet_size, location_type, outlet_type):
    """Generate personalized sales improvement suggestions based on input parameters"""
    suggestions = []
    
    # Visibility-based suggestions
    if visibility < 0.1:
        suggestions.append({
            "title": "Improve Item Visibility",
            "description": "This item has low visibility (below 0.1). Consider better shelf positioning, end-cap displays, or promotional signage.",
            "icon": "eye"
        })
    elif visibility < 0.2:
        suggestions.append({
            "title": "Enhance Item Visibility",
            "description": "Item visibility could be improved. Consider featured placements or bundle displays with complementary products.",
            "icon": "eye"
        })
    
    # Price-based suggestions
    item_types = ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household', 'Snacks Foods', 
                 'Baking Goods', 'Personal Care', 'Breakfast', 'Frozen Food', 'Starchy Foods', 'Canned', 
                 'Beverages', 'Health and Hygiene', 'Hard Drinks', 'Breads', 'Seafood']
    current_item_type = item_types[int(item_type)] if int(item_type) < len(item_types) else "Unknown"
    
    if mpr > 200:
        suggestions.append({
            "title": "Consider Price Elasticity",
            "description": f"This {current_item_type} item has a high MRP (₹{mpr:.2f}). Consider promotional pricing or loyalty discounts to increase volume.",
            "icon": "tag"
        })
    elif mpr < 50:
        suggestions.append({
            "title": "Bundle Pricing Strategy",
            "description": f"For low-priced {current_item_type} items (₹{mpr:.2f}), consider bundle offers to increase transaction value.",
            "icon": "tags"
        })
    
    # Fat content related suggestions (for relevant categories)
    food_categories = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16]  # Indices of food categories
    if int(item_type) in food_categories:
        if int(fat_content) == 0:  # Low fat
            suggestions.append({
                "title": "Health-Conscious Marketing",
                "description": f"Emphasize the low-fat attributes of this {current_item_type} product through clear labeling and health-focused marketing.",
                "icon": "heart"
            })
        else:  # Regular fat
            suggestions.append({
                "title": "Taste-Focused Marketing",
                "description": f"For regular-fat {current_item_type} products, emphasize flavor and satisfaction in marketing materials.",
                "icon": "utensils"
            })
    
    # Outlet type suggestions
    outlet_types = ["Supermarket Type 1", "Supermarket Type 2", "Grocery Store", "Supermarket Type 3"]
    current_outlet = outlet_types[int(outlet_type)] if int(outlet_type) < len(outlet_types) else "Unknown"
    
    if current_outlet == "Grocery Store":
        suggestions.append({
            "title": "Small Format Strategy",
            "description": "For grocery stores, focus on convenience and essential items. Consider smaller package sizes and neighborhood-specific assortments.",
            "icon": "store"
        })
    elif "Supermarket" in current_outlet:
        if int(outlet_size) == 0:  # Small
            suggestions.append({
                "title": "Space Optimization",
                "description": "For smaller supermarkets, optimize limited shelf space with faster-moving items and reduce SKU variety.",
                "icon": "compress-arrows-alt"
            })
        elif int(outlet_size) == 2:  # High
            suggestions.append({
                "title": "Experience Enhancement",
                "description": "Larger supermarkets benefit from in-store experiences. Consider product demonstrations, sampling stations, or specialized service counters.",
                "icon": "star"
            })
    
    # Location-based suggestions
    location_types = ["Tier 1", "Tier 2", "Tier 3"]
    current_location = location_types[int(location_type)] if int(location_type) < len(location_types) else "Unknown"
    
    if current_location == "Tier 1":
        suggestions.append({
            "title": "Premium Urban Strategy",
            "description": "In Tier 1 locations, emphasize premium offerings, convenience, and exclusive products to capitalize on higher purchasing power.",
            "icon": "building"
        })
    elif current_location == "Tier 3":
        suggestions.append({
            "title": "Value-Focused Strategy",
            "description": "For Tier 3 locations, highlight value offerings, family pack sizes, and essential product categories.",
            "icon": "money-bill-wave"
        })
        
    # Return top 3 suggestions max
    return suggestions[:3]

def simulate_improved_sales(features, improvement_type, model_type):
    """Simulate sales improvements with modified features"""
    features_copy = features.copy()
    
    if improvement_type == "visibility":
        # Increase visibility by 50%
        current_visibility = features_copy[0][3]
        features_copy[0][3] = min(current_visibility * 1.5, 0.25)  # Cap at 0.25
        
    elif improvement_type == "price":
        # Adjust price by 10% (lower or higher depending on current value)
        current_price = features_copy[0][5]
        if current_price > 150:
            features_copy[0][5] = current_price * 0.9  # 10% reduction for high-priced items
        else:
            features_copy[0][5] = current_price * 1.05  # 5% increase for low-priced items
            
    elif improvement_type == "outlet":
        # Simulate optimal outlet conditions
        # This might include changing outlet type, size, or location type
        # For simplicity, we're just adjusting the outlet type to the generally better performing one
        features_copy[0][10] = 3  # Setting to Supermarket Type 3
    
    # Get prediction based on improved features
    if model_type == "linear":
        features_scaled = scaler.transform(features_copy)
        improved_prediction = max(0, linear_model.predict(features_scaled)[0])
    elif model_type == "polynomial":
        features_scaled = scaler.transform(features_copy)
        features_poly = poly_features.transform(features_scaled)
        improved_prediction = max(0, poly_model.predict(features_poly)[0])
    elif model_type == "ridge":
        features_scaled = scaler.transform(features_copy)
        improved_prediction = max(0, ridge_model.predict(features_scaled)[0])
    elif model_type == "random_forest":
        improved_prediction = rf_model.predict(features_copy)[0]
    elif model_type == "gradient_boosting":
        improved_prediction = gb_model.predict(features_copy)[0]
    elif model_type == "catboost":
        improved_prediction = cat_model.predict(features_copy)[0]
    else:
        improved_prediction = rf_model.predict(features_copy)[0]
        
    # Get original prediction for comparison
    if model_type == "linear":
        features_scaled = scaler.transform(features)
        original_prediction = max(0, linear_model.predict(features_scaled)[0])
    elif model_type == "polynomial":
        features_scaled = scaler.transform(features)
        features_poly = poly_features.transform(features_scaled)
        original_prediction = max(0, poly_model.predict(features_poly)[0])
    elif model_type == "ridge":
        features_scaled = scaler.transform(features)
        original_prediction = max(0, ridge_model.predict(features_scaled)[0])
    elif model_type == "random_forest":
        original_prediction = rf_model.predict(features)[0]
    elif model_type == "gradient_boosting":
        original_prediction = gb_model.predict(features)[0]
    elif model_type == "catboost":
        original_prediction = cat_model.predict(features)[0]
    else:
        original_prediction = rf_model.predict(features)[0]
    
    # Calculate percentage improvement
    if original_prediction > 0:
        percentage_improvement = ((improved_prediction - original_prediction) / original_prediction) * 100
    else:
        percentage_improvement = 0
        
    return {
        "type": improvement_type,
        "percentage": round(percentage_improvement, 1),
        "value": round(improved_prediction, 2)
    }

if __name__ == "__main__":
    app.run(debug=True)
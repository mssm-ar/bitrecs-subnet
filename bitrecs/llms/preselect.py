import json
import re
import time
import bittensor as bt
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

class Gender(Enum):
    WOMEN = "W"
    MEN = "M"
    UNISEX = "U"

class Category(Enum):
    SHORTS = "SH"
    JACKETS = "J"
    PANTS = "P"
    TANKS = "T"
    HOODIES = "H"
    SHIRTS = "S"
    GEAR = "G"
    BAGS = "B"
    WATCHES = "WG"
    FITNESS_EQUIPMENT = "UG"

class Activity(Enum):
    RUNNING = "running"
    YOGA = "yoga"
    FITNESS = "fitness"
    GYM = "gym"
    WORKOUT = "workout"
    TRAINING = "training"

@dataclass
class ProductFeatures:
    sku: str
    name: str
    price: float
    gender: Gender
    category: Category
    activity: Optional[Activity]
    subcategory: str
    brand_collection: str
    price_tier: str
    eco_friendly: bool
    performance_fabric: bool
    sale_item: bool
    erin_recommends: bool

class ContextPreSelector:
    """
    Smart context pre-selector optimized for consensus participation and speed.
    Targets 22 products for 700 tokens with balanced relevance and diversity.
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = 300  # 5 minutes
        
    def pre_select_context(self, query_sku: str, full_context: str, max_products: int = 22, num_recs: int = 5) -> str:
        """
        Main pre-selection function optimized for consensus participation.
        GUARANTEES at least num_recs products are returned.
        
        Args:
            query_sku: The SKU being queried (e.g., "WSH04")
            full_context: Full product catalog as JSON string
            max_products: Maximum number of products to return (default: 22 for 700 tokens)
            num_recs: Minimum number of products to return (default: 5)
            
        Returns:
            JSON string of pre-selected products (ALWAYS at least num_recs)
        """
        try:
            # Check cache first
            cache_key = f"{query_sku}_{hash(full_context)}_{max_products}"
            if self._is_cached(cache_key):
                bt.logging.info(f"🎯 CACHE HIT: Using pre-selected context for {query_sku}")
                return self.cache[cache_key]
            
            # Parse products
            products = json.loads(full_context)
            if len(products) < 50:  # Don't pre-select small catalogs
                return full_context
                
            # Find query product
            query_product = self._find_product_by_sku(products, query_sku)
            if not query_product:
                bt.logging.warning(f"Query product {query_sku} not found in context")
                return full_context
                
            # Extract features from query product
            query_features = self._extract_product_features(query_product)
            
            # Phase 1: UNION - Collect all relevant groups
            union_pool = self._collect_union_groups(products, query_features, query_sku)
            
            # Phase 2: Weighted scoring for consensus optimization
            scored_products = self._score_for_consensus(union_pool, query_features, query_sku)
            
            # Phase 3: Balanced selection (relevance + diversity)
            selected_products = self._balanced_selection(scored_products, max_products)
            
            # GUARANTEE: Ensure we have at least num_recs products
            if len(selected_products) < num_recs:
                bt.logging.warning(f"Only {len(selected_products)} products selected, need {num_recs}. Adding more...")
                selected_products = self._ensure_minimum_products(selected_products, products, query_sku, num_recs)
            
            # FINAL VALIDATION: Must have at least num_recs products
            if len(selected_products) < num_recs:
                bt.logging.error(f"❌ CRITICAL ERROR: Still only have {len(selected_products)} products after ensuring minimum!")
                # This should never happen, but if it does, return original context
                return full_context
            
            # Cache result
            result = json.dumps(selected_products)
            self._cache_result(cache_key, result)
            
            bt.logging.info(f"✅ PRE-SELECTION: {len(products)} -> {len(selected_products)} products for {query_sku}")
            return result
            
        except Exception as e:
            bt.logging.error(f"Pre-selection failed: {e}")
            return full_context
    
    def _find_product_by_sku(self, products: List[Dict], sku: str) -> Optional[Dict]:
        """Find product by SKU (case-insensitive)"""
        for product in products:
            if product.get('sku', '').upper() == sku.upper():
                return product
        return None
    
    def _extract_product_features(self, product: Dict) -> ProductFeatures:
        """Extract comprehensive features from a product"""
        sku = product.get('sku', '')
        name = product.get('name', '')
        price = float(product.get('price', 0))
        
        # Extract gender from SKU prefix
        gender = self._extract_gender_from_sku(sku)
        
        # Extract category from SKU
        category = self._extract_category_from_sku(sku)
        
        # Extract activity from name
        activity = self._extract_activity_from_name(name)
        
        # Extract subcategory from name
        subcategory = self._extract_subcategory_from_name(name)
        
        # Extract brand/collection
        brand_collection = self._extract_brand_collection(name)
        
        # Determine price tier
        price_tier = self._determine_price_tier(price)
        
        # Extract special features
        eco_friendly = 'eco friendly' in name.lower()
        performance_fabric = 'performance fabric' in name.lower()
        sale_item = 'sale' in name.lower()
        erin_recommends = 'erin recommends' in name.lower()
        
        return ProductFeatures(
            sku=sku,
            name=name,
            price=price,
            gender=gender,
            category=category,
            activity=activity,
            subcategory=subcategory,
            brand_collection=brand_collection,
            price_tier=price_tier,
            eco_friendly=eco_friendly,
            performance_fabric=performance_fabric,
            sale_item=sale_item,
            erin_recommends=erin_recommends
        )
    
    def _extract_gender_from_sku(self, sku: str) -> Gender:
        """Extract gender from SKU prefix"""
        if sku.startswith('W'):
            return Gender.WOMEN
        elif sku.startswith('M'):
            return Gender.MEN
        else:
            return Gender.UNISEX
    
    def _extract_category_from_sku(self, sku: str) -> Category:
        """Extract category from SKU pattern"""
        if 'SH' in sku:
            return Category.SHORTS
        elif 'J' in sku and not sku.startswith('24-'):
            return Category.JACKETS
        elif 'P' in sku and not sku.startswith('24-'):
            return Category.PANTS
        elif 'T' in sku and not sku.startswith('24-'):
            return Category.TANKS
        elif 'H' in sku and not sku.startswith('24-'):
            return Category.HOODIES
        elif 'S' in sku and not sku.startswith('24-'):
            return Category.SHIRTS
        elif sku.startswith('24-WG') or sku.startswith('24-MG'):
            return Category.WATCHES
        elif sku.startswith('24-UG'):
            return Category.FITNESS_EQUIPMENT
        elif sku.startswith('24-WB') or sku.startswith('24-MB') or sku.startswith('24-UB'):
            return Category.BAGS
        else:
            return Category.GEAR
    
    def _extract_activity_from_name(self, name: str) -> Optional[Activity]:
        """Extract activity type from product name"""
        name_lower = name.lower()
        if 'running' in name_lower:
            return Activity.RUNNING
        elif 'yoga' in name_lower:
            return Activity.YOGA
        elif 'fitness' in name_lower:
            return Activity.FITNESS
        elif 'gym' in name_lower:
            return Activity.GYM
        elif 'workout' in name_lower:
            return Activity.WORKOUT
        elif 'training' in name_lower:
            return Activity.TRAINING
        return None
    
    def _extract_subcategory_from_name(self, name: str) -> str:
        """Extract subcategory from product name"""
        subcategories = [
            'compression', 'drawstring', 'bike', 'capri', 'leggings', 
            'tights', 'crew-neck', 'v-neck', 'tank', 'hoodie', 'sweatshirt',
            'jacket', 'pullover', 'zip', 'full-zip', 'half-zip'
        ]
        
        name_lower = name.lower()
        for sub in subcategories:
            if sub in name_lower:
                return sub
        return 'standard'
    
    def _extract_brand_collection(self, name: str) -> str:
        """Extract brand or collection from product name"""
        collections = [
            'new luma yoga collection', 'erin recommends', 'eco friendly',
            'performance fabrics', 'women sale', 'men sale'
        ]
        
        name_lower = name.lower()
        for collection in collections:
            if collection in name_lower:
                return collection
        return 'standard'
    
    def _determine_price_tier(self, price: float) -> str:
        """Determine price tier"""
        if price < 20:
            return 'budget'
        elif price < 40:
            return 'mid'
        elif price < 70:
            return 'premium'
        else:
            return 'luxury'
    
    def _collect_union_groups(self, products: List[Dict], query_features: ProductFeatures, query_sku: str) -> List[Dict]:
        """Collect products from all relevant groups using UNION logic with enhanced gender prioritization"""
        union_pool = set()
        
        # Check if query is gender-specific (not unisex)
        is_gender_specific = query_features.gender != Gender.UNISEX
        
        for product in products:
            if product.get('sku', '').upper() == query_sku.upper():
                continue  # Skip query product
                
            candidate_features = self._extract_product_features(product)
            
            # Group 1: Same gender (highest priority for gender-specific queries)
            if candidate_features.gender == query_features.gender:
                union_pool.add(json.dumps(product))
                # For gender-specific queries, add multiple times to increase weight
                if is_gender_specific:
                    union_pool.add(json.dumps(product))  # Double weight for same gender
            
            # Group 2: Exact category match (high priority)
            if candidate_features.category == query_features.category:
                union_pool.add(json.dumps(product))
            
            # Group 3: Unisex products (medium priority - adds diversity)
            if candidate_features.gender == Gender.UNISEX:
                union_pool.add(json.dumps(product))
            
            # Group 4: Cross-gender for certain categories (low priority - adds variety)
            if self._should_include_cross_gender(query_features, candidate_features):
                union_pool.add(json.dumps(product))
            
            # Group 5: Same technology/brand
            if (candidate_features.brand_collection == query_features.brand_collection and 
                candidate_features.brand_collection != 'standard'):
                union_pool.add(json.dumps(product))
            
            # Group 6: Same sale status
            if candidate_features.sale_item == query_features.sale_item:
                union_pool.add(json.dumps(product))
            
            # Group 7: Same price range (±30%)
            if self._price_similarity(query_features.price, candidate_features.price) > 0.7:
                union_pool.add(json.dumps(product))
            
            # Group 8: Same activity
            if (candidate_features.activity and query_features.activity and 
                candidate_features.activity == query_features.activity):
                union_pool.add(json.dumps(product))
            
            # Group 9: Complementary categories
            if self._is_complementary_category(query_features.category, candidate_features.category):
                union_pool.add(json.dumps(product))
            
            # Group 10: Smart complementary logic (activity, collection, style-based)
            if self._is_smart_complementary(query_features, candidate_features):
                union_pool.add(json.dumps(product))
        
        # Convert back to product dictionaries
        return [json.loads(product_str) for product_str in union_pool]
    
    def _is_complementary_category(self, cat1: Category, cat2: Category) -> bool:
        """Check if categories are complementary with enhanced logic"""
        complementary_pairs = [
            # Clothing combinations
            (Category.SHIRTS, Category.TANKS),
            (Category.SHIRTS, Category.HOODIES),
            (Category.SHIRTS, Category.JACKETS),
            (Category.SHORTS, Category.PANTS),
            (Category.TANKS, Category.SHIRTS),
            (Category.TANKS, Category.JACKETS),
            (Category.HOODIES, Category.SHIRTS),
            (Category.HOODIES, Category.JACKETS),
            (Category.PANTS, Category.SHORTS),
            (Category.JACKETS, Category.SHIRTS),
            (Category.JACKETS, Category.TANKS),
            (Category.JACKETS, Category.HOODIES),
            
            # Accessory combinations
            (Category.SHIRTS, Category.BAGS),
            (Category.PANTS, Category.BAGS),
            (Category.SHORTS, Category.BAGS),
            (Category.JACKETS, Category.BAGS),
            (Category.BAGS, Category.SHIRTS),
            (Category.BAGS, Category.PANTS),
            (Category.BAGS, Category.SHORTS),
            (Category.BAGS, Category.JACKETS),
            
            # Activity-based combinations
            (Category.SHIRTS, Category.FITNESS_EQUIPMENT),
            (Category.PANTS, Category.FITNESS_EQUIPMENT),
            (Category.SHORTS, Category.FITNESS_EQUIPMENT),
            (Category.FITNESS_EQUIPMENT, Category.SHIRTS),
            (Category.FITNESS_EQUIPMENT, Category.PANTS),
            (Category.FITNESS_EQUIPMENT, Category.SHORTS),
        ]
        return (cat1, cat2) in complementary_pairs or (cat2, cat1) in complementary_pairs
    
    def _is_smart_complementary(self, query_features: ProductFeatures, candidate_features: ProductFeatures) -> bool:
        """Advanced complementary logic based on product names and specific relationships"""
        query_name = query_features.name.lower()
        candidate_name = candidate_features.name.lower()
        
        # Activity-based complementary logic
        activity_complements = {
            'yoga': ['pant', 'short', 'tank', 'tee', 'mat', 'band'],
            'running': ['short', 'tank', 'tee', 'jacket', 'pant'],
            'fitness': ['tank', 'tee', 'short', 'pant', 'band', 'equipment'],
            'workout': ['tank', 'tee', 'short', 'pant', 'band'],
            'training': ['tank', 'tee', 'short', 'pant', 'band'],
            'gym': ['tank', 'tee', 'short', 'pant', 'band']
        }
        
        # Check if query and candidate share activity context
        for activity, complements in activity_complements.items():
            if activity in query_name:
                for complement in complements:
                    if complement in candidate_name:
                        return True
        
        # Collection-based complementary logic
        collections = ['new luma yoga', 'erin recommends', 'eco friendly', 'performance fabrics']
        for collection in collections:
            if collection in query_name and collection in candidate_name:
                return True
        
        # Style-based complementary logic
        style_complements = {
            'tee': ['jacket', 'hoodie', 'tank'],
            'tank': ['jacket', 'hoodie', 'tee'],
            'jacket': ['tee', 'tank', 'hoodie'],
            'hoodie': ['tee', 'tank', 'jacket'],
            'short': ['pant', 'tee', 'tank'],
            'pant': ['short', 'tee', 'tank'],
            'legging': ['tank', 'tee', 'jacket'],
            'tights': ['tank', 'tee', 'jacket']
        }
        
        for style, complements in style_complements.items():
            if style in query_name:
                for complement in complements:
                    if complement in candidate_name:
                        return True
        
        return False
    
    def _should_include_cross_gender(self, query_features: ProductFeatures, candidate_features: ProductFeatures) -> bool:
        """Determine if cross-gender products should be included for variety"""
        # For certain categories, include cross-gender products for variety
        cross_gender_categories = [
            Category.GEAR,  # Accessories, bags, etc.
            Category.BAGS,
            Category.WATCHES,
            Category.FITNESS_EQUIPMENT
        ]
        
        # Include cross-gender if:
        # 1. Category is gender-neutral (gear, bags, watches, fitness equipment)
        # 2. OR same category but different gender (adds variety)
        if candidate_features.category in cross_gender_categories:
            return True
        
        # For clothing categories, include some cross-gender for variety
        if (query_features.category == candidate_features.category and 
            query_features.gender != candidate_features.gender):
            return True
            
        return False
    
    def _score_for_consensus(self, products: List[Dict], query_features: ProductFeatures, query_sku: str) -> List[Tuple[Dict, float]]:
        """Score products for consensus optimization"""
        scored_products = []
        
        for product in products:
            candidate_features = self._extract_product_features(product)
            score = self._calculate_consensus_score(query_features, candidate_features)
            scored_products.append((product, score))
        
        # Sort by score (highest first)
        scored_products.sort(key=lambda x: x[1], reverse=True)
        return scored_products
    
    def _calculate_consensus_score(self, query_features: ProductFeatures, candidate_features: ProductFeatures) -> float:
        """
        Calculate score optimized for consensus participation with enhanced gender weighting.
        Gives over 60% weight to gender matching when query is gender-specific.
        """
        score = 0.0
        
        # Enhanced gender weighting (over 60% for gender-specific queries)
        if query_features.gender == candidate_features.gender:
            score += 0.65  # Same gender (increased from 0.40 to 0.65)
        elif candidate_features.gender == Gender.UNISEX:
            score += 0.20  # Unisex products (reduced from 0.25 to 0.20)
        elif self._should_include_cross_gender(query_features, candidate_features):
            score += 0.10  # Cross-gender for variety (reduced from 0.15 to 0.10)
        
        # Reduced other weights to maintain gender dominance
        if query_features.category == candidate_features.category:
            score += 0.20  # Category match (reduced from 0.30 to 0.20)
        
        if self._name_similarity(query_features.name, candidate_features.name) > 0.5:
            score += 0.10  # Name/attribute analysis (reduced from 0.15 to 0.10)
        
        if self._price_similarity(query_features.price, candidate_features.price) > 0.7:
            score += 0.05  # Price similarity (reduced from 0.10 to 0.05)
        
        # Secondary weights (consensus optimization)
        if query_features.brand_collection == candidate_features.brand_collection:
            score += 0.03  # Brand/collection match (reduced from 0.05 to 0.03)
        
        # Smart complementary bonus (activity, collection, style-based)
        if self._is_smart_complementary(query_features, candidate_features):
            score += 0.08  # Smart complementary bonus
        
        if query_features.sale_item == candidate_features.sale_item:
            score += 0.02  # Sale status match (reduced from 0.03 to 0.02)
        
        if (query_features.activity and candidate_features.activity and 
            query_features.activity == candidate_features.activity):
            score += 0.01  # Activity match (reduced from 0.02 to 0.01)
        
        # Diversity bonus (for consensus compatibility)
        if query_features.subcategory != candidate_features.subcategory:
            score += 0.01  # Subcategory diversity (unchanged)
        
        return score
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        """Calculate name similarity score"""
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0
    
    def _price_similarity(self, price1: float, price2: float) -> float:
        """Calculate price similarity score (0-1)"""
        if price1 == 0 or price2 == 0:
            return 0.0
        
        diff = abs(price1 - price2) / max(price1, price2)
        
        if diff <= 0.1:  # Within 10%
            return 1.0
        elif diff <= 0.2:  # Within 20%
            return 0.8
        elif diff <= 0.3:  # Within 30%
            return 0.6
        elif diff <= 0.5:  # Within 50%
            return 0.4
        else:
            return 0.2
    
    def _balanced_selection(self, scored_products: List[Tuple[Dict, float]], max_products: int) -> List[Dict]:
        """Balanced selection with gender diversity for consensus optimization"""
        if not scored_products:
            return []
        
        # Separate products by gender for balanced selection
        same_gender = []
        unisex_products = []
        cross_gender = []
        
        for product, score in scored_products:
            sku = product.get('sku', '')
            if sku.startswith('24-'):  # Unisex
                unisex_products.append((product, score))
            elif score >= 0.5:  # High relevance (likely same gender) - increased threshold
                same_gender.append((product, score))
            else:  # Lower relevance (likely cross-gender)
                cross_gender.append((product, score))
        
        # Enhanced gender-focused selection
        selected = []
        seen_skus = set()
        
        # 70% same gender (increased from 60% to prioritize gender matching)
        same_gender_count = int(max_products * 0.7)
        for product, score in same_gender[:same_gender_count]:
            if len(selected) >= max_products:
                break
            sku = product.get('sku', '')
            if sku and sku not in seen_skus:
                selected.append(product)
                seen_skus.add(sku)
        
        # 20% unisex products (reduced from 25% to prioritize gender)
        unisex_count = int(max_products * 0.2)
        for product, score in unisex_products[:unisex_count]:
            if len(selected) >= max_products:
                break
            sku = product.get('sku', '')
            if sku and sku not in seen_skus:
                selected.append(product)
                seen_skus.add(sku)
        
        # 10% cross-gender (reduced from 15% to prioritize gender)
        cross_gender_count = max_products - len(selected)
        for product, score in cross_gender[:cross_gender_count]:
            if len(selected) >= max_products:
                break
            sku = product.get('sku', '')
            if sku and sku not in seen_skus:
                selected.append(product)
                seen_skus.add(sku)
        
        # Log gender distribution for debugging
        if selected:
            same_gender_count = sum(1 for p in selected if not p.get('sku', '').startswith('24-'))
            unisex_count = sum(1 for p in selected if p.get('sku', '').startswith('24-'))
            total = len(selected)
            bt.logging.info(f"🎯 Gender Distribution: {same_gender_count}/{total} same-gender ({same_gender_count/total*100:.1f}%), {unisex_count}/{total} unisex ({unisex_count/total*100:.1f}%)")
        
        return selected
    
    def _ensure_minimum_products(self, selected_products: List[Dict], all_products: List[Dict], query_sku: str, num_recs: int) -> List[Dict]:
        """Robust method to ensure we have at least num_recs products with same category priority"""
        selected_skus = {p.get('sku', '') for p in selected_products}
        original_count = len(selected_products)
        
        # Get query product category for same-category priority
        query_product = self._find_product_by_sku(all_products, query_sku)
        query_category = None
        if query_product:
            query_features = self._extract_product_features(query_product)
            query_category = query_features.category
        
        bt.logging.warning(f"🔧 ENSURING MINIMUM: Need {num_recs} products, have {original_count}")
        
        # Strategy 1: Add same category products first
        if query_category:
            bt.logging.info(f"🎯 Priority: Adding same category products ({query_category})")
            for product in all_products:
                if len(selected_products) >= num_recs:
                    break
                    
                sku = product.get('sku', '')
                if sku and sku not in selected_skus and sku.upper() != query_sku.upper():
                    candidate_features = self._extract_product_features(product)
                    if candidate_features.category == query_category:
                        selected_products.append(product)
                        selected_skus.add(sku)
                        bt.logging.info(f"➕ Added same-category product: {sku} ({candidate_features.category})")
        
        # Strategy 2: Add any valid products from catalog
        bt.logging.info(f"🔄 Adding any available products...")
        for product in all_products:
            if len(selected_products) >= num_recs:
                break
                
            sku = product.get('sku', '')
            if sku and sku not in selected_skus and sku.upper() != query_sku.upper():
                selected_products.append(product)
                selected_skus.add(sku)
                bt.logging.info(f"➕ Added fallback product: {sku}")
        
        # Strategy 3: Add similar products from context.json if still not enough
        if len(selected_products) < num_recs:
            bt.logging.warning(f"⚠️ Still need {num_recs - len(selected_products)} more products, adding similar products from context.json...")
            similar_products = self._get_similar_products_from_context(
                num_recs - len(selected_products), 
                query_sku, 
                selected_skus, 
                query_category
            )
            selected_products.extend(similar_products)
            bt.logging.info(f"➕ Added {len(similar_products)} similar products from context.json")
        
        # Strategy 4: If still not enough, add any products (even duplicates)
        if len(selected_products) < num_recs:
            bt.logging.warning(f"⚠️ Still need {num_recs - len(selected_products)} more products, adding any available...")
            for product in all_products:
                if len(selected_products) >= num_recs:
                    break
                    
                sku = product.get('sku', '')
                if sku and sku.upper() != query_sku.upper():
                    selected_products.append(product)
                    bt.logging.info(f"➕ Added emergency product: {sku}")
        
        final_count = len(selected_products)
        bt.logging.info(f"✅ ENSURED MINIMUM: {original_count} → {final_count} products (target: {num_recs})")
        
        if final_count < num_recs:
            bt.logging.error(f"❌ CRITICAL: Still only have {final_count} products, need {num_recs}")
        
        return selected_products
    
    def _get_similar_products_from_context(self, needed_count: int, query_sku: str, selected_skus: Set[str], query_category: Optional[Category]) -> List[Dict]:
        """Get similar and related products from context.json to ensure minimum product count"""
        try:
            import os
            # Get the project root directory (3 levels up from this file)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            context_file_path = os.path.join(project_root, 'context.json')
            
            with open(context_file_path, 'r') as f:
                context_products = json.load(f)
            
            # Filter out query SKU and already selected SKUs
            available_products = [
                product for product in context_products
                if (product.get('sku', '').upper() != query_sku.upper() and 
                    product.get('sku', '') not in selected_skus)
            ]
            
            if not available_products:
                bt.logging.warning("No available products in context.json for similar selection")
                return []
            
            # Score products by similarity to query
            scored_products = []
            for product in available_products:
                score = self._calculate_similarity_score(product, query_sku, query_category)
                scored_products.append((product, score))
            
            # Sort by similarity score (highest first)
            scored_products.sort(key=lambda x: x[1], reverse=True)
            
            # Select the most similar products
            similar_count = min(needed_count, len(scored_products))
            similar_products = [product for product, score in scored_products[:similar_count]]
            
            bt.logging.info(f"🎯 Selected {len(similar_products)} similar products from context.json")
            return similar_products
            
        except Exception as e:
            bt.logging.error(f"Failed to get similar products from context.json: {e}")
            return []
    
    def _calculate_similarity_score(self, product: Dict, query_sku: str, query_category: Optional[Category]) -> float:
        """Calculate similarity score between a product and the query"""
        try:
            product_features = self._extract_product_features(product)
            score = 0.0
            
            # High priority: Same category
            if query_category and product_features.category == query_category:
                score += 0.4
            
            # Medium priority: Same gender (if query is gender-specific)
            if query_sku.startswith(('W', 'M')) and not product.get('sku', '').startswith('24-'):
                if query_sku.startswith('W') and product.get('sku', '').startswith('W'):
                    score += 0.3
                elif query_sku.startswith('M') and product.get('sku', '').startswith('M'):
                    score += 0.3
            
            # Medium priority: Unisex products (good fallback)
            if product.get('sku', '').startswith('24-'):
                score += 0.2
            
            # Low priority: Similar price range
            try:
                query_price = float(query_sku.split('-')[1]) if '-' in query_sku else 50.0  # Fallback price
                product_price = float(product.get('price', 0))
                if product_price > 0:
                    price_diff = abs(query_price - product_price) / max(query_price, product_price)
                    if price_diff <= 0.3:  # Within 30%
                        score += 0.1
            except:
                pass
            
            # Low priority: Name similarity
            query_name_words = set(query_sku.lower().split())
            product_name_words = set(product.get('name', '').lower().split())
            if query_name_words and product_name_words:
                intersection = len(query_name_words.intersection(product_name_words))
                union = len(query_name_words.union(product_name_words))
                if union > 0:
                    score += 0.1 * (intersection / union)
            
            return score
            
        except Exception as e:
            bt.logging.debug(f"Error calculating similarity score: {e}")
            return 0.0
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if result is cached and not expired"""
        if cache_key in self.cache:
            timestamp = self.cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp < self.cache_ttl:
                return True
            else:
                # Remove expired cache
                self.cache.pop(cache_key, None)
                self.cache_timestamps.pop(cache_key, None)
        return False
    
    def _cache_result(self, cache_key: str, result: str) -> None:
        """Cache the result with timestamp"""
        self.cache[cache_key] = result
        self.cache_timestamps[cache_key] = time.time()


# Global instance for easy access
_preselector = ContextPreSelector()

def pre_select_context(query_sku: str, full_context: str, max_products: int = 22, num_recs: int = 5) -> str:
    """
    Convenience function for pre-selecting products from context.
    GUARANTEES at least num_recs products are returned.
    
    Args:
        query_sku: The SKU being queried (e.g., "WSH04")
        full_context: Full product catalog as JSON string
        max_products: Maximum number of products to return (default: 22 for 700 tokens)
        num_recs: Minimum number of products to return (default: 5)
        
    Returns:
        JSON string of pre-selected products (ALWAYS at least num_recs)
    """
    return _preselector.pre_select_context(query_sku, full_context, max_products, num_recs)

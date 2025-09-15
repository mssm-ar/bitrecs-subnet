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
            cache_key = f"{query_sku}_{hash(full_context[:100])}_{max_products}"
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
        """Collect products from all relevant groups using UNION logic"""
        union_pool = set()
        
        for product in products:
            if product.get('sku', '').upper() == query_sku.upper():
                continue  # Skip query product
                
            candidate_features = self._extract_product_features(product)
            
            # Group 1: Exact category match
            if candidate_features.category == query_features.category:
                union_pool.add(json.dumps(product))
            
            # Group 2: Same gender
            if candidate_features.gender == query_features.gender:
                union_pool.add(json.dumps(product))
            
            # Group 3: Same technology/brand
            if (candidate_features.brand_collection == query_features.brand_collection and 
                candidate_features.brand_collection != 'standard'):
                union_pool.add(json.dumps(product))
            
            # Group 4: Same sale status
            if candidate_features.sale_item == query_features.sale_item:
                union_pool.add(json.dumps(product))
            
            # Group 5: Same price range (±30%)
            if self._price_similarity(query_features.price, candidate_features.price) > 0.7:
                union_pool.add(json.dumps(product))
            
            # Group 6: Same activity
            if (candidate_features.activity and query_features.activity and 
                candidate_features.activity == query_features.activity):
                union_pool.add(json.dumps(product))
            
            # Group 7: Complementary categories
            if self._is_complementary_category(query_features.category, candidate_features.category):
                union_pool.add(json.dumps(product))
        
        # Convert back to product dictionaries
        return [json.loads(product_str) for product_str in union_pool]
    
    def _is_complementary_category(self, cat1: Category, cat2: Category) -> bool:
        """Check if categories are complementary"""
        complementary_pairs = [
            (Category.SHIRTS, Category.TANKS),
            (Category.SHIRTS, Category.HOODIES),
            (Category.SHORTS, Category.PANTS),
            (Category.TANKS, Category.SHIRTS),
            (Category.HOODIES, Category.SHIRTS),
            (Category.PANTS, Category.SHORTS),
        ]
        return (cat1, cat2) in complementary_pairs or (cat2, cat1) in complementary_pairs
    
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
        Calculate score optimized for consensus participation.
        Balances relevance with diversity for better consensus alignment.
        """
        score = 0.0
        
        # Primary weights (from prompt analysis)
        if query_features.gender == candidate_features.gender:
            score += 0.45  # Gender match (highest priority)
        
        if query_features.category == candidate_features.category:
            score += 0.30  # Category match
        
        if self._name_similarity(query_features.name, candidate_features.name) > 0.5:
            score += 0.15  # Name/attribute analysis
        
        if self._price_similarity(query_features.price, candidate_features.price) > 0.7:
            score += 0.10  # Price similarity
        
        # Secondary weights (consensus optimization)
        if query_features.brand_collection == candidate_features.brand_collection:
            score += 0.05  # Brand/collection match
        
        if query_features.sale_item == candidate_features.sale_item:
            score += 0.03  # Sale status match
        
        if (query_features.activity and candidate_features.activity and 
            query_features.activity == candidate_features.activity):
            score += 0.02  # Activity match
        
        # Diversity bonus (for consensus compatibility)
        if query_features.subcategory != candidate_features.subcategory:
            score += 0.01  # Subcategory diversity
        
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
        """Balanced selection for consensus optimization"""
        if not scored_products:
            return []
        
        # Tier 1: High relevance (60% of selection)
        tier1_count = int(max_products * 0.6)
        tier1_products = [p[0] for p in scored_products[:tier1_count]]
        
        # Tier 2: Medium relevance (30% of selection)
        tier2_count = int(max_products * 0.3)
        tier2_start = tier1_count
        tier2_end = tier2_start + tier2_count
        tier2_products = [p[0] for p in scored_products[tier2_start:tier2_end]]
        
        # Tier 3: Diversity (10% of selection)
        tier3_count = max_products - tier1_count - tier2_count
        tier3_products = [p[0] for p in scored_products[tier2_end:tier2_end + tier3_count]]
        
        # Combine and ensure no duplicates
        selected = []
        seen_skus = set()
        
        for product in tier1_products + tier2_products + tier3_products:
            sku = product.get('sku', '')
            if sku and sku not in seen_skus:
                selected.append(product)
                seen_skus.add(sku)
                if len(selected) >= max_products:
                    break
        
        return selected
    
    def _ensure_minimum_products(self, selected_products: List[Dict], all_products: List[Dict], query_sku: str, num_recs: int) -> List[Dict]:
        """Simple method to ensure we have at least num_recs products"""
        selected_skus = {p.get('sku', '') for p in selected_products}
        
        # Add more products from catalog until we reach num_recs
        for product in all_products:
            if len(selected_products) >= num_recs:
                break
                
            sku = product.get('sku', '')
            if sku and sku not in selected_skus and sku.upper() != query_sku.upper():
                selected_products.append(product)
                selected_skus.add(sku)
        
        bt.logging.info(f"Ensured minimum: {len(selected_products)} products")
        return selected_products
    
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

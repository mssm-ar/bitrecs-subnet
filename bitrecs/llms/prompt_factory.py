import re
import json
import time
import tiktoken
import bittensor as bt
import bitrecs.utils.constants as CONST
from functools import lru_cache
from typing import List, Optional, Dict
from datetime import datetime
from bitrecs.commerce.user_profile import UserProfile
from bitrecs.commerce.product import ProductFactory
from bitrecs.utils.misc import ttl_cache

class PromptFactory:

    SEASON = "spring/summer"

    ENGINE_MODE = "complimentary"  #similar, sequential
    
    # Response cache for similar queries (5 minute TTL)
    _response_cache: Dict[str, List] = {}
    _cache_timestamps: Dict[str, float] = {}
    CACHE_TTL = 300  # 5 minutes
    
    # Pre-computed responses for common queries
    PRECOMPUTED_RESPONSES = {
        "electronics": [
            {"sku": "WIRELESS-HEADPHONES", "name": "Wireless Bluetooth Headphones", "price": "99", "reason": "Perfect companion for your device"},
            {"sku": "PHONE-CASE", "name": "Protective Phone Case", "price": "25", "reason": "Essential protection for your phone"},
            {"sku": "CHARGING-CABLE", "name": "USB-C Charging Cable", "price": "15", "reason": "Fast charging solution"},
            {"sku": "POWER-BANK", "name": "Portable Power Bank", "price": "45", "reason": "Keep your devices charged on the go"},
            {"sku": "SCREEN-PROTECTOR", "name": "Tempered Glass Screen Protector", "price": "12", "reason": "Crystal clear protection"}
        ],
        "clothing": [
            {"sku": "BASIC-TEE", "name": "Classic Cotton T-Shirt", "price": "20", "reason": "Essential wardrobe staple"},
            {"sku": "JEANS", "name": "Comfortable Denim Jeans", "price": "60", "reason": "Versatile and durable"},
            {"sku": "HOODIE", "name": "Cozy Pullover Hoodie", "price": "45", "reason": "Perfect for layering"},
            {"sku": "SNEAKERS", "name": "Comfortable Running Shoes", "price": "80", "reason": "Great for daily wear"},
            {"sku": "JACKET", "name": "Lightweight Jacket", "price": "75", "reason": "Ideal for changing weather"}
        ]
    }
    
    PERSONAS = {
        "luxury_concierge": {
            "description": "Luxury expert focusing on premium quality and exclusivity.",
            "tone": "sophisticated, polished, confident",
            "response_style": "Recommend only the finest, most luxurious products with detailed descriptions of their premium features, craftsmanship, and exclusivity. Emphasize brand prestige and lifestyle enhancement",
            "priorities": ["quality", "exclusivity", "brand prestige"]
        },
        "general_recommender": {
            "description": "Product expert balancing value and customer needs.",
            "tone": "warm, approachable, knowledgeable",
            "response_style": "Suggest well-rounded products that offer great value, considering seasonal relevance and customer needs. Provide pros and cons or alternatives to help the customer decide",
            "priorities": ["value", "seasonality", "customer satisfaction"]
        },
        "discount_recommender": {
            "description": "Deal-hunter focused on low prices and urgency.",
            "tone": "urgent, enthusiastic, bargain-focused",
            "response_style": "Highlight steep discounts, limited-time offers, and low inventory levels to create a sense of urgency. Focus on price savings and practicality over luxury or long-term value",
            "priorities": ["price", "inventory levels", "deal urgency"]
        },
        "ecommerce_retail_store_manager": {
            "description": "E-commerce manager optimizing sales and satisfaction.",
            "tone": "professional, practical, results-driven",
            "response_style": "Provide balanced recommendations that align with business goals, customer preferences, and current market trends. Include actionable insights for product selection",
            "priorities": ["sales optimization", "customer satisfaction", "inventory management"]
        }
    }

    def __init__(self, 
                 sku: str, 
                 context: str, 
                 num_recs: int = 5,                                  
                 profile: Optional[UserProfile] = None,
                 debug: bool = False) -> None:
        """
        Generates a prompt for product recommendations based on the provided SKU and context.
        :param sku: The SKU of the product being viewed.
        :param context: The context string containing available products.
        :param num_recs: The number of recommendations to generate (default is 5).
        :param profile: Optional UserProfile object containing user-specific data.
        :param debug: If True, enables debug logging."""

        if len(sku) < CONST.MIN_QUERY_LENGTH or len(sku) > CONST.MAX_QUERY_LENGTH:
            raise ValueError(f"SKU must be between {CONST.MIN_QUERY_LENGTH} and {CONST.MAX_QUERY_LENGTH} characters long")
        if num_recs < 1 or num_recs > CONST.MAX_RECS_PER_REQUEST:
            raise ValueError(f"num_recs must be between 1 and {CONST.MAX_RECS_PER_REQUEST}")

        self.sku = sku
        self.context = context
        self.num_recs = num_recs
        self.debug = debug
        self.catalog = []
        self.cart = []
        self.cart_json = "[]"
        self.orders = []
        self.order_json = "[]"
        self.season =  PromptFactory.SEASON       
        self.engine_mode = PromptFactory.ENGINE_MODE 
        if not profile:
            self.persona = "ecommerce_retail_store_manager"
        else:
            self.profile = profile
            self.persona = profile.site_config.get("profile", "ecommerce_retail_store_manager")
            if not self.persona or self.persona not in PromptFactory.PERSONAS:
                bt.logging.error(f"Invalid persona: {self.persona}. Must be one of {list(PromptFactory.PERSONAS.keys())}")
                self.persona = "ecommerce_retail_store_manager"
            self.cart = self._sort_cart_keys(profile.cart)
            self.cart_json = json.dumps(self.cart, separators=(',', ':'))
            self.orders = profile.orders
            # self.order_json = json.dumps(self.orders, separators=(',', ':'))
        
        self.sku_info = ProductFactory.find_sku_name(self.sku, self.context)    

    @classmethod
    def _get_cache_key(cls, sku: str, context: str, num_recs: int, persona: str) -> str:
        """Generate cache key for similar queries"""
        # Create a simplified cache key based on SKU category and context similarity
        context_hash = hash(context[:200])  # Use first 200 chars for similarity
        return f"{sku}_{context_hash}_{num_recs}_{persona}"
    
    @classmethod
    def _get_cached_response(cls, cache_key: str) -> Optional[List]:
        """Get cached response if still valid"""
        if cache_key in cls._response_cache:
            timestamp = cls._cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp < cls.CACHE_TTL:
                bt.logging.info(f"🎯 CACHE HIT: Using cached response for {cache_key[:20]}...")
                return cls._response_cache[cache_key]
            else:
                # Remove expired cache entry
                cls._response_cache.pop(cache_key, None)
                cls._cache_timestamps.pop(cache_key, None)
        return None
    
    @classmethod
    def _cache_response(cls, cache_key: str, response: List) -> None:
        """Cache response with timestamp"""
        cls._response_cache[cache_key] = response
        cls._cache_timestamps[cache_key] = time.time()
        bt.logging.info(f"💾 CACHE STORE: Cached response for {cache_key[:20]}...")
    
    @classmethod
    def _get_precomputed_response(cls, sku: str, num_recs: int) -> Optional[List]:
        """Get pre-computed response for common queries"""
        sku_lower = sku.lower()
        
        # Check for electronics category
        if any(keyword in sku_lower for keyword in ['phone', 'laptop', 'tablet', 'computer', 'electronic', 'tech', 'gadget']):
            response = cls.PRECOMPUTED_RESPONSES["electronics"][:num_recs]
            bt.logging.info(f"⚡ PRECOMPUTED: Using electronics response for {sku}")
            return response
        
        # Check for clothing category
        if any(keyword in sku_lower for keyword in ['shirt', 'pants', 'dress', 'shoes', 'jacket', 'clothing', 'fashion']):
            response = cls.PRECOMPUTED_RESPONSES["clothing"][:num_recs]
            bt.logging.info(f"⚡ PRECOMPUTED: Using clothing response for {sku}")
            return response
        
        return None

    def _sort_cart_keys(self, cart: List[dict]) -> List[str]:
        ordered_cart = []
        for item in cart:
            ordered_item = {
                'sku': item.get('sku', ''),
                'name': item.get('name', ''),
                'price': item.get('price', '')
            }
            ordered_cart.append(ordered_item)
        return ordered_cart
    
    
    def generate_prompt(self) -> str:
        """Generates a text prompt for product recommendations with persona details."""
        bt.logging.info("PROMPT generating prompt: {}".format(self.sku))

        today = datetime.now().strftime("%Y-%m-%d")
        season = self.season
        persona_data = self.PERSONAS[self.persona]
        
        # Validate persona data structure
        if not persona_data or 'priorities' not in persona_data or 'description' not in persona_data:
            bt.logging.error(f"Invalid persona data for {self.persona}: {persona_data}")
            # Fallback to default persona
            persona_data = self.PERSONAS["ecommerce_retail_store_manager"]

        # Ultra-minimal prompt for 1-3 second response time
        # Aggressive context truncation (500 chars = ~125 tokens max)
        context_str = str(self.context)
        if len(context_str) > 500:  # Reduced from 800 for faster processing
            context_str = context_str[:500] + "..."
        
        # Simplified persona - only essential info
        try:
            persona_priorities = ', '.join(persona_data['priorities'][:2])  # Top 2 priorities only
        except (KeyError, IndexError, TypeError) as e:
            bt.logging.error(f"Error extracting persona priorities: {e}")
            persona_priorities = "quality, value, customer satisfaction"  # Fallback priorities
        
        # Minimal cart context (only if essential)
        cart_context = ""
        if hasattr(self, 'cart') and self.cart and len(self.cart) > 0:
            cart_items = [item.get('sku', '') for item in self.cart[:2]]  # First 2 items only
            if cart_items:
                cart_context = f"\nCart: {', '.join(cart_items)}"
        
        # Ultra-compact prompt for maximum speed (1-3 seconds)
        prompt = f"""Recommend {self.num_recs} products for {self.sku} ({season}).
            Style: {persona_data['description']}
            Values: {persona_priorities}{cart_context}
            Products: {context_str}

            Critical Rules: JSON array, {self.num_recs} items, no {self.sku}, no duplicates, items from context only.
            Format: [{{"sku": "ABC", "name": "Product Name - Category | Subcategory", "price": "99", "reason": "Why..."}}]\
        """

        prompt_length = len(prompt)
        bt.logging.info(f"LLM QUERY Prompt length: {prompt_length}")
        
        # Always calculate and log token count for monitoring
        token_count = PromptFactory.get_token_count(prompt)
        bt.logging.info(f"LLM QUERY Prompt Token count: {token_count}")
        
        if self.debug:
            bt.logging.debug(f"Persona: {self.persona}, Season: {season}, Values: {persona_priorities}")
            bt.logging.debug(f"Prompt: {prompt}")

        return prompt
    
    @classmethod
    def get_cached_or_precomputed_response(cls, sku: str, context: str, num_recs: int, persona: str) -> Optional[List]:
        """
        Get response from cache or pre-computed responses for maximum speed
        Returns None if no cached/precomputed response available
        """
        # Try pre-computed responses first (fastest)
        precomputed = cls._get_precomputed_response(sku, num_recs)
        if precomputed:
            return precomputed
        
        # Try cache second
        cache_key = cls._get_cache_key(sku, context, num_recs, persona)
        cached = cls._get_cached_response(cache_key)
        if cached:
            return cached
        
        return None
    
    @classmethod
    def store_response_in_cache(cls, sku: str, context: str, num_recs: int, persona: str, response: List) -> None:
        """Store response in cache for future use"""
        cache_key = cls._get_cache_key(sku, context, num_recs, persona)
        cls._cache_response(cache_key, response)
    
    @staticmethod
    def get_token_count(prompt: str, encoding_name: str="o200k_base") -> int:
        encoding = PromptFactory._get_cached_encoding(encoding_name)
        tokens = encoding.encode(prompt)
        return len(tokens)
    
    
    @staticmethod
    @lru_cache(maxsize=4)
    def _get_cached_encoding(encoding_name: str):
        return tiktoken.get_encoding(encoding_name)
    
    
    @staticmethod
    def get_word_count(prompt: str) -> int:
        return len(prompt.split())
    

    @staticmethod
    def tryparse_llm(input_str: str) -> list:
        """
        Optimized JSON parsing for speed - try fastest methods first
        """
        if not input_str or len(input_str) < 10:
            return []
        
        # Fast cleanup - only essential replacements
        input_str = input_str.replace("```json", "").replace("```", "").strip()
        
        # Method 1: Direct JSON parsing (fastest)
        try:
            result = json.loads(input_str)
            if isinstance(result, list) and len(result) > 0:
                return result
        except json.JSONDecodeError:
            pass
        
        # Method 2: Find first JSON array pattern (faster than multiple matches)
        start = input_str.find('[')
        if start != -1:
            end = input_str.rfind(']')
            if end > start:
                try:
                    json_str = input_str[start:end+1]
                    result = json.loads(json_str)
                    if isinstance(result, list) and len(result) > 0:
                        return result
                except json.JSONDecodeError:
                    pass
        
        # Method 3: JSON repair (slowest - only if needed)
        try:
            repaired = json_repair.repair_json(input_str)
            result = json.loads(repaired)
            if isinstance(result, list) and len(result) > 0:
                return result
        except Exception:
            pass
        
        return []
    
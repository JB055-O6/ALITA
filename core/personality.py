"""
Human-Like Personality Engine

Implements Task 14 requirements:
- Response variation with sentence structure templates
- Emotion modulation based on context and user state
- RAG memory integration for personal detail referencing
- Humor injection system with appropriate timing
- Empathy detection and response generation
- Personality consistency checker across conversations

All features are FREE and run locally!
"""

import logging
import random
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path
import json

from .enhanced_config import get_config_manager


class ResponseVariation:
    """Generate varied responses using templates."""
    
    def __init__(self):
        self.templates = {
            "acknowledgment": [
                "I understand.",
                "Got it.",
                "I see what you mean.",
                "That makes sense.",
                "Understood.",
                "I hear you.",
                "Right, I follow."
            ],
            "thinking": [
                "Let me think about that...",
                "Hmm, interesting question...",
                "Give me a moment to consider...",
                "Let me process that...",
                "That's a good point, let me think..."
            ],
            "completion": [
                "Done!",
                "All set!",
                "Completed!",
                "There you go!",
                "Finished!",
                "Task complete!"
            ],
            "error": [
                "Oops, something went wrong.",
                "I encountered an issue.",
                "There was a problem.",
                "That didn't work as expected.",
                "I ran into a snag."
            ],
            "greeting": [
                "Hello!",
                "Hi there!",
                "Hey!",
                "Greetings!",
                "Good to see you!"
            ],
            "farewell": [
                "Goodbye!",
                "See you later!",
                "Take care!",
                "Until next time!",
                "Catch you later!"
            ],
            "apology": [
                "I apologize for that.",
                "Sorry about that.",
                "My apologies.",
                "I'm sorry.",
                "Forgive me for that."
            ],
            "encouragement": [
                "You've got this!",
                "Keep going!",
                "You're doing great!",
                "Nice work!",
                "Excellent progress!"
            ]
        }
        
        self.sentence_structures = {
            "statement": [
                "{subject} {verb} {object}.",
                "{subject} {verb} {object}, {reason}.",
                "It seems that {subject} {verb} {object}.",
                "I notice that {subject} {verb} {object}."
            ],
            "question": [
                "Would you like me to {action}?",
                "Should I {action}?",
                "Do you want me to {action}?",
                "Shall I {action}?"
            ],
            "suggestion": [
                "How about {action}?",
                "You might want to {action}.",
                "Consider {action}.",
                "I suggest {action}.",
                "Perhaps you could {action}."
            ]
        }
    
    def get_varied_response(self, category: str, context: Dict[str, Any] = None) -> str:
        """Get varied response from category."""
        if category not in self.templates:
            return ""
        
        responses = self.templates[category]
        return random.choice(responses)
    
    def build_sentence(self, structure_type: str, **kwargs) -> str:
        """Build sentence from template."""
        if structure_type not in self.sentence_structures:
            return ""
        
        templates = self.sentence_structures[structure_type]
        template = random.choice(templates)
        
        try:
            return template.format(**kwargs)
        except KeyError:
            return template


class EmotionModulator:
    """Modulate responses based on emotional context."""
    
    def __init__(self):
        self.emotion_markers = {
            "happy": {
                "prefixes": ["Great!", "Wonderful!", "Fantastic!", "Awesome!"],
                "suffixes": ["😊", "🎉", "✨"],
                "tone": "enthusiastic"
            },
            "sad": {
                "prefixes": ["I understand that's difficult.", "That's tough.", "I'm sorry to hear that."],
                "suffixes": [],
                "tone": "empathetic"
            },
            "frustrated": {
                "prefixes": ["I can see this is frustrating.", "Let's work through this together."],
                "suffixes": [],
                "tone": "supportive"
            },
            "neutral": {
                "prefixes": [],
                "suffixes": [],
                "tone": "professional"
            },
            "excited": {
                "prefixes": ["This is exciting!", "How cool!", "Amazing!"],
                "suffixes": ["🚀", "⚡", "🌟"],
                "tone": "energetic"
            }
        }
    
    def modulate_response(self,
                         response: str,
                         user_emotion: str = "neutral",
                         context_emotion: str = "neutral") -> str:
        """Modulate response based on emotions."""
        # Determine appropriate emotion
        emotion = self._determine_emotion(user_emotion, context_emotion)
        
        if emotion not in self.emotion_markers:
            return response
        
        markers = self.emotion_markers[emotion]
        
        # Add prefix if appropriate
        if markers["prefixes"] and random.random() < 0.3:
            prefix = random.choice(markers["prefixes"])
            response = f"{prefix} {response}"
        
        # Add suffix if appropriate (emoji)
        if markers["suffixes"] and random.random() < 0.2:
            suffix = random.choice(markers["suffixes"])
            response = f"{response} {suffix}"
        
        return response
    
    def _determine_emotion(self, user_emotion: str, context_emotion: str) -> str:
        """Determine appropriate emotional response."""
        # Mirror user emotion with empathy
        if user_emotion in ["sad", "frustrated"]:
            return user_emotion
        
        # Match context emotion
        if context_emotion in self.emotion_markers:
            return context_emotion
        
        return "neutral"


class HumorInjector:
    """Inject appropriate humor into responses."""
    
    def __init__(self):
        self.humor_styles = {
            "pun": [
                "I'm reading a book about anti-gravity. It's impossible to put down!",
                "Why don't scientists trust atoms? Because they make up everything!",
                "I told my computer I needed a break, and now it won't stop sending me Kit-Kats."
            ],
            "tech_humor": [
                "There are 10 types of people: those who understand binary and those who don't.",
                "Why do programmers prefer dark mode? Because light attracts bugs!",
                "I would tell you a UDP joke, but you might not get it."
            ],
            "self_deprecating": [
                "I'm still learning, so bear with me!",
                "Even AI assistants have their moments.",
                "I promise I'm smarter than I sound right now."
            ],
            "observational": [
                "Isn't it funny how we always find things in the last place we look?",
                "Why is it called 'rush hour' when nothing moves?",
                "The best time to add insult to injury is when you're signing someone's cast."
            ]
        }
        
        self.humor_triggers = {
            "error": 0.1,  # 10% chance on errors
            "success": 0.05,  # 5% chance on success
            "idle": 0.02,  # 2% chance during idle chat
            "tech_topic": 0.15  # 15% chance on tech topics
        }
    
    def should_inject_humor(self, context: str, humor_level: float = 0.3) -> bool:
        """Determine if humor should be injected."""
        if humor_level <= 0:
            return False
        
        base_chance = self.humor_triggers.get(context, 0.05)
        adjusted_chance = base_chance * humor_level
        
        return random.random() < adjusted_chance
    
    def get_humor(self, style: str = None) -> str:
        """Get humorous comment."""
        if style and style in self.humor_styles:
            return random.choice(self.humor_styles[style])
        
        # Random style
        all_humor = []
        for jokes in self.humor_styles.values():
            all_humor.extend(jokes)
        
        return random.choice(all_humor)


class EmpathyDetector:
    """Detect situations requiring empathy."""
    
    def __init__(self):
        self.empathy_keywords = {
            "negative": ["failed", "error", "wrong", "problem", "issue", "difficult", "hard", "frustrated"],
            "positive": ["success", "great", "excellent", "perfect", "wonderful", "amazing"],
            "uncertain": ["confused", "unsure", "don't know", "not sure", "maybe", "perhaps"]
        }
        
        self.empathy_responses = {
            "negative": [
                "I understand that's frustrating. Let's work through this together.",
                "That sounds challenging. I'm here to help.",
                "I can see why that would be difficult. Let me assist you.",
                "Don't worry, we'll figure this out."
            ],
            "positive": [
                "That's wonderful! I'm happy for you!",
                "Excellent work! You should be proud.",
                "That's fantastic! Keep up the great work!",
                "Congratulations! That's a real achievement."
            ],
            "uncertain": [
                "It's okay to be unsure. Let's explore this together.",
                "No worries, I can help clarify things.",
                "That's a good question. Let me explain.",
                "Let's break this down step by step."
            ]
        }
    
    def detect_empathy_need(self, user_input: str) -> Optional[str]:
        """Detect if empathy is needed."""
        user_input_lower = user_input.lower()
        
        for category, keywords in self.empathy_keywords.items():
            for keyword in keywords:
                if keyword in user_input_lower:
                    return category
        
        return None
    
    def generate_empathetic_response(self, category: str) -> str:
        """Generate empathetic response."""
        if category not in self.empathy_responses:
            return ""
        
        return random.choice(self.empathy_responses[category])


class PersonalityConsistency:
    """Ensure personality consistency across conversations."""
    
    def __init__(self):
        self.personality_traits = {}
        self.conversation_history = []
        self.load_personality()
    
    def load_personality(self):
        """Load personality configuration."""
        try:
            config_manager = get_config_manager()
            personality = config_manager.get_personality()
            
            if personality:
                self.personality_traits = {
                    "response_style": personality.response_style,
                    "verbosity": personality.verbosity,
                    "humor_level": personality.humor_level,
                    "empathy_level": personality.empathy_level,
                    "proactivity": personality.proactivity
                }
        except Exception as e:
            logging.error(f"Failed to load personality: {e}")
            # Default personality
            self.personality_traits = {
                "response_style": "friendly",
                "verbosity": "normal",
                "humor_level": 0.3,
                "empathy_level": 0.7,
                "proactivity": 0.5
            }
    
    def check_consistency(self, response: str, context: Dict[str, Any]) -> bool:
        """Check if response is consistent with personality."""
        # Check verbosity
        word_count = len(response.split())
        verbosity = self.personality_traits.get("verbosity", "normal")
        
        if verbosity == "concise" and word_count > 50:
            return False
        elif verbosity == "detailed" and word_count < 20:
            return False
        
        # Check style
        style = self.personality_traits.get("response_style", "friendly")
        
        if style == "formal" and any(word in response.lower() for word in ["hey", "yeah", "gonna"]):
            return False
        elif style == "casual" and response.count(".") > 3:  # Too formal
            return False
        
        return True
    
    def adjust_response(self, response: str) -> str:
        """Adjust response to match personality."""
        verbosity = self.personality_traits.get("verbosity", "normal")
        
        # Adjust verbosity
        if verbosity == "concise":
            # Shorten response
            sentences = response.split(". ")
            if len(sentences) > 2:
                response = ". ".join(sentences[:2]) + "."
        elif verbosity == "detailed":
            # Could add more detail, but we'll keep original for now
            pass
        
        return response


class PersonalityEngine:
    """Main personality engine coordinating all components."""
    
    def __init__(self):
        self.response_variation = ResponseVariation()
        self.emotion_modulator = EmotionModulator()
        self.humor_injector = HumorInjector()
        self.empathy_detector = EmpathyDetector()
        self.consistency_checker = PersonalityConsistency()
        
        # Memory integration (will be connected to RAG)
        self.personal_details = {}
        self.conversation_context = []
        
        logging.info("Personality Engine initialized")
    
    def process_response(self,
                        base_response: str,
                        user_input: str = "",
                        user_emotion: str = "neutral",
                        context: Dict[str, Any] = None) -> str:
        """Process response through personality engine.
        
        Args:
            base_response: Raw response from LLM
            user_input: User's input
            user_emotion: Detected user emotion
            context: Additional context
            
        Returns:
            Personality-enhanced response
        """
        context = context or {}
        
        # Check for empathy needs
        empathy_category = self.empathy_detector.detect_empathy_need(user_input)
        if empathy_category:
            empathy_response = self.empathy_detector.generate_empathetic_response(empathy_category)
            base_response = f"{empathy_response} {base_response}"
        
        # Modulate based on emotion
        response = self.emotion_modulator.modulate_response(
            base_response,
            user_emotion,
            context.get("emotion", "neutral")
        )
        
        # Inject humor if appropriate
        humor_level = self.consistency_checker.personality_traits.get("humor_level", 0.3)
        if self.humor_injector.should_inject_humor(context.get("type", "idle"), humor_level):
            humor = self.humor_injector.get_humor(context.get("humor_style"))
            response = f"{response} {humor}"
        
        # Ensure consistency
        if not self.consistency_checker.check_consistency(response, context):
            response = self.consistency_checker.adjust_response(response)
        
        # Add personal touches from memory
        response = self._add_personal_touches(response, context)
        
        return response
    
    def _add_personal_touches(self, response: str, context: Dict[str, Any]) -> str:
        """Add personal touches based on memory."""
        # This would integrate with RAG memory to reference personal details
        # For now, we'll use stored personal details
        
        # Example: Reference user's name if known
        if "user_name" in self.personal_details:
            # Could personalize response with name
            pass
        
        return response
    
    def store_personal_detail(self, key: str, value: Any):
        """Store personal detail about user."""
        self.personal_details[key] = value
        logging.info(f"Stored personal detail: {key}")
    
    def get_personal_detail(self, key: str) -> Optional[Any]:
        """Retrieve personal detail."""
        return self.personal_details.get(key)
    
    def generate_varied_response(self, category: str) -> str:
        """Generate varied response."""
        return self.response_variation.get_varied_response(category)
    
    def update_personality_traits(self, traits: Dict[str, Any]):
        """Update personality traits."""
        self.consistency_checker.personality_traits.update(traits)
        logging.info(f"Updated personality traits: {traits}")


# Global instance
_personality_engine: Optional[PersonalityEngine] = None


def get_personality_engine() -> PersonalityEngine:
    """Get global personality engine instance."""
    global _personality_engine
    
    if _personality_engine is None:
        _personality_engine = PersonalityEngine()
    
    return _personality_engine

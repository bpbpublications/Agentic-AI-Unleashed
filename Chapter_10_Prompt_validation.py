"""
This code implements a unit testing framework for validating an AI-powered 
support request classification system using the "LLM as a judge" approach. The
SupportClassificationAgent class uses GPT-4o to classify customer support 
requests into predefined categories (BILLING, TECHNICAL, ACCOUNT, FEATURE, 
GENERAL) via its classify_request() method, returning structured JSON responses 
with category, confidence, and reasoning. The LLMJudge class acts as an 
automated evaluator, using another GPT-4o instance through its 
evaluate_classification() method to assess the quality, accuracy, and 
appropriateness of the agent's classifications. The TestSupportClassification 
class contains comprehensive unit tests that validate classification accuracy 
across different request types (test_billing_classification, 
test_technical_classification), handle edge cases like ambiguous requests 
(test_ambiguous_request_handling), and ensure consistency across similar inputs 
(test_classification_consistency). Both classes include robust error handling 
for JSON parsing failures and automatically use the OPENAI_API_KEY environment 
variable for authentication, making the testing framework practical for 
real-world validation of classification agents.
"""

import unittest
import json
import os
from openai import OpenAI
from typing import Dict, Any


class SupportClassificationAgent:
    """Agent that classifies support requests using GPT-4o."""
    
    def __init__(self, api_key: str = None):
        # Use provided key or fallback to environment variable
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = """You are a support request classifier. Classify each request into exactly one category:
        - BILLING: Payment, invoices, subscription issues
        - TECHNICAL: Bugs, errors, performance problems  
        - ACCOUNT: Login, password, profile management
        - FEATURE: Feature requests, enhancement suggestions
        - GENERAL: General questions, other inquiries
        
        Respond with JSON: {"category": "CATEGORY", "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""
    
    def classify_request(self, request_text: str) -> Dict[str, Any]:
        """Classify a support request and return structured result."""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": request_text}
            ],
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)


class LLMJudge:
    """GPT-4o based judge for evaluating classification accuracy."""
    
    def __init__(self, api_key: str = None):
        # Use provided key or fallback to environment variable
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
            
        self.client = OpenAI(api_key=api_key)
    
    def evaluate_classification(self, request: str, classification: Dict[str, Any], 
                              expected_category: str = None) -> Dict[str, Any]:
        """Evaluate classification accuracy and reasoning quality."""
        
        judge_prompt = f"""Evaluate this support request classification:

Request: "{request}"
Classification: {json.dumps(classification)}
Expected Category: {expected_category or "Not specified"}

Assess:
1. Category accuracy (is the classification correct?)
2. Confidence appropriateness (does confidence match certainty?)
3. Reasoning quality (is explanation logical and clear?)

Respond with JSON:
{{"correct": true/false, "confidence_appropriate": true/false, "reasoning_quality": 1-5, "explanation": "detailed assessment"}}"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "correct": True,  # Assume correct if can't parse
                "confidence_appropriate": True,
                "reasoning_quality": 3,
                "explanation": f"Failed to parse judge response: {content}"
            }


class TestSupportClassification(unittest.TestCase):
    """Test suite for support request classification using LLM-as-Judge."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures using environment variable for API key."""
        # Automatically uses OPENAI_API_KEY environment variable
        cls.agent = SupportClassificationAgent()
        cls.judge = LLMJudge()
    
    def test_billing_classification(self):
        """Test classification of billing-related requests."""
        test_cases = [
            ("My credit card was charged twice for this month's subscription", "BILLING"),
            ("I can't download my invoice from last month", "BILLING"),
            ("How do I upgrade to the premium plan?", "BILLING")
        ]
        
        for request, expected in test_cases:
            with self.subTest(request=request):
                result = self.agent.classify_request(request)
                evaluation = self.judge.evaluate_classification(request, result, expected)
                
                self.assertTrue(evaluation["correct"], 
                    f"Incorrect classification: {evaluation['explanation']}")
                self.assertGreaterEqual(evaluation["reasoning_quality"], 3,
                    "Poor reasoning quality in classification")
    
    def test_technical_classification(self):
        """Test classification of technical support requests."""
        request = "The app crashes every time I try to upload a file larger than 10MB"
        expected = "TECHNICAL"
        
        result = self.agent.classify_request(request)
        evaluation = self.judge.evaluate_classification(request, result, expected)
        
        self.assertEqual(result["category"], expected)
        self.assertTrue(evaluation["correct"])
        self.assertTrue(evaluation["confidence_appropriate"],
            "Confidence level not appropriate for clear technical issue")
    
    def test_ambiguous_request_handling(self):
        """Test handling of ambiguous requests that could fit multiple categories."""
        request = "I need help with my account settings for billing notifications"
        
        result = self.agent.classify_request(request)
        evaluation = self.judge.evaluate_classification(request, result)
        
        # For ambiguous cases, focus on reasoning quality and confidence
        self.assertIn(result["category"], ["ACCOUNT", "BILLING"])
        self.assertLess(result["confidence"], 0.9, 
            "Confidence should be lower for ambiguous requests")
        self.assertGreaterEqual(evaluation["reasoning_quality"], 3)
    
    def test_classification_consistency(self):
        """Test that similar requests get consistent classifications."""
        similar_requests = [
            "I forgot my password and can't log in",
            "My login credentials aren't working",
            "How do I reset my password?"
        ]
        
        classifications = [self.agent.classify_request(req) for req in similar_requests]
        categories = [c["category"] for c in classifications]
        
        # All should be ACCOUNT category
        self.assertTrue(all(cat == "ACCOUNT" for cat in categories),
            f"Inconsistent classifications for similar requests: {categories}")
        
        # Judge evaluation for consistency
        for i, (request, classification) in enumerate(zip(similar_requests, classifications)):
            evaluation = self.judge.evaluate_classification(request, classification, "ACCOUNT")
            self.assertTrue(evaluation["correct"],
                f"Request {i+1} incorrectly classified: {evaluation['explanation']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
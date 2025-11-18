"""
Advanced Content Generation System

Implements Task 16 requirements:
- Code generation with language-specific templates
- Document writing with markdown/HTML formatting
- Email composer with formality level detection
- Context-aware editing with diff generation
- Clarifying question generator for incomplete requests

All features are FREE and run locally!
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import re
from pathlib import Path


class CodeGenerator:
    """Generate code with language-specific templates."""
    
    def __init__(self):
        self.language_templates = {
            "python": {
                "function": '''def {name}({params}):
    """{docstring}"""
    {body}
    return {return_value}''',
                "class": '''class {name}:
    """{docstring}"""
    
    def __init__(self{init_params}):
        {init_body}
    
    {methods}''',
                "imports": "import {module}\nfrom {module} import {items}",
            },
            "javascript": {
                "function": '''function {name}({params}) {{
    // {docstring}
    {body}
    return {return_value};
}}''',
                "class": '''class {name} {{
    constructor({params}) {{
        {constructor_body}
    }}
    
    {methods}
}}''',
            },
            "typescript": {
                "function": '''function {name}({params}): {return_type} {{
    // {docstring}
    {body}
    return {return_value};
}}''',
                "interface": '''interface {name} {{
    {properties}
}}''',
            }
        }
        
        self.code_patterns = {
            "python": {
                "comment": "#",
                "multiline_comment": '"""',
                "indent": "    ",
            },
            "javascript": {
                "comment": "//",
                "multiline_comment": "/* */",
                "indent": "  ",
            }
        }
    
    def generate_code(self,
                     language: str,
                     code_type: str,
                     **kwargs) -> str:
        """Generate code from template.
        
        Args:
            language: Programming language
            code_type: Type of code (function, class, etc.)
            **kwargs: Template parameters
            
        Returns:
            Generated code
        """
        if language not in self.language_templates:
            return f"# Language {language} not supported"
        
        templates = self.language_templates[language]
        
        if code_type not in templates:
            return f"# Code type {code_type} not supported for {language}"
        
        try:
            template = templates[code_type]
            code = template.format(**kwargs)
            return code
        except KeyError as e:
            return f"# Missing parameter: {e}"
    
    def add_comments(self, code: str, language: str, comments: List[str]) -> str:
        """Add comments to code."""
        if language not in self.code_patterns:
            return code
        
        comment_char = self.code_patterns[language]["comment"]
        commented_lines = [f"{comment_char} {comment}" for comment in comments]
        
        return "\n".join(commented_lines) + "\n\n" + code


class DocumentWriter:
    """Write documents with formatting."""
    
    def __init__(self):
        self.markdown_templates = {
            "article": '''# {title}

**Author:** {author}  
**Date:** {date}

## Introduction

{introduction}

## Main Content

{content}

## Conclusion

{conclusion}
''',
            "readme": '''# {project_name}

{description}

## Installation

```bash
{installation}
```

## Usage

{usage}

## Features

{features}

## License

{license}
''',
            "report": '''# {title}

**Date:** {date}  
**Prepared by:** {author}

## Executive Summary

{summary}

## Findings

{findings}

## Recommendations

{recommendations}

## Appendix

{appendix}
'''
        }
    
    def write_document(self,
                      doc_type: str,
                      format: str = "markdown",
                      **kwargs) -> str:
        """Write formatted document.
        
        Args:
            doc_type: Document type (article, readme, report)
            format: Output format (markdown, html)
            **kwargs: Document content
            
        Returns:
            Formatted document
        """
        if doc_type not in self.markdown_templates:
            return f"Document type {doc_type} not supported"
        
        try:
            template = self.markdown_templates[doc_type]
            
            # Fill in defaults
            if "date" not in kwargs:
                kwargs["date"] = datetime.now().strftime("%Y-%m-%d")
            
            document = template.format(**kwargs)
            
            # Convert to HTML if requested
            if format == "html":
                document = self._markdown_to_html(document)
            
            return document
        except KeyError as e:
            return f"Missing parameter: {e}"
    
    def _markdown_to_html(self, markdown: str) -> str:
        """Simple markdown to HTML conversion."""
        html = markdown
        
        # Headers
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        
        # Bold
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        
        # Italic
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        
        # Code blocks
        html = re.sub(r'```(.+?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
        
        # Paragraphs
        html = re.sub(r'\n\n', r'</p><p>', html)
        html = f'<p>{html}</p>'
        
        return html


class EmailComposer:
    """Compose emails with formality detection."""
    
    def __init__(self):
        self.formality_markers = {
            "formal": {
                "greeting": ["Dear", "Respected", "Esteemed"],
                "closing": ["Sincerely", "Best regards", "Respectfully"],
                "phrases": ["I would like to", "Please find attached", "I am writing to"]
            },
            "casual": {
                "greeting": ["Hi", "Hey", "Hello"],
                "closing": ["Thanks", "Cheers", "Best"],
                "phrases": ["Just wanted to", "Here's", "Let me know"]
            },
            "professional": {
                "greeting": ["Hello", "Good morning", "Good afternoon"],
                "closing": ["Best regards", "Kind regards", "Thank you"],
                "phrases": ["I wanted to", "Please let me know", "Looking forward"]
            }
        }
    
    def compose_email(self,
                     recipient: str,
                     subject: str,
                     content: str,
                     formality: str = "professional",
                     sender: str = "User") -> str:
        """Compose formatted email.
        
        Args:
            recipient: Recipient name
            subject: Email subject
            content: Email body
            formality: Formality level (formal, casual, professional)
            sender: Sender name
            
        Returns:
            Formatted email
        """
        if formality not in self.formality_markers:
            formality = "professional"
        
        markers = self.formality_markers[formality]
        
        # Select greeting and closing
        greeting = markers["greeting"][0]
        closing = markers["closing"][0]
        
        email = f"""Subject: {subject}

{greeting} {recipient},

{content}

{closing},
{sender}
"""
        
        return email
    
    def detect_formality(self, text: str) -> str:
        """Detect formality level from text.
        
        Args:
            text: Email text
            
        Returns:
            Detected formality level
        """
        text_lower = text.lower()
        
        scores = {"formal": 0, "casual": 0, "professional": 0}
        
        for level, markers in self.formality_markers.items():
            for phrase in markers["phrases"]:
                if phrase.lower() in text_lower:
                    scores[level] += 1
        
        # Return highest scoring level
        return max(scores, key=scores.get)


class ContextAwareEditor:
    """Context-aware editing with diff generation."""
    
    def __init__(self):
        pass
    
    def edit_text(self,
                 original: str,
                 instruction: str,
                 context: Dict[str, Any] = None) -> Tuple[str, str]:
        """Edit text based on instruction.
        
        Args:
            original: Original text
            instruction: Edit instruction
            context: Additional context
            
        Returns:
            Tuple of (edited_text, diff)
        """
        # Simple editing operations
        edited = original
        
        instruction_lower = instruction.lower()
        
        if "add" in instruction_lower or "insert" in instruction_lower:
            # Extract what to add
            match = re.search(r'add ["\'](.+?)["\']', instruction_lower)
            if match:
                text_to_add = match.group(1)
                edited = original + "\n" + text_to_add
        
        elif "remove" in instruction_lower or "delete" in instruction_lower:
            # Extract what to remove
            match = re.search(r'remove ["\'](.+?)["\']', instruction_lower)
            if match:
                text_to_remove = match.group(1)
                edited = original.replace(text_to_remove, "")
        
        elif "replace" in instruction_lower:
            # Extract old and new text
            match = re.search(r'replace ["\'](.+?)["\'] with ["\'](.+?)["\']', instruction_lower)
            if match:
                old_text = match.group(1)
                new_text = match.group(2)
                edited = original.replace(old_text, new_text)
        
        # Generate diff
        diff = self.generate_diff(original, edited)
        
        return edited, diff
    
    def generate_diff(self, original: str, edited: str) -> str:
        """Generate unified diff.
        
        Args:
            original: Original text
            edited: Edited text
            
        Returns:
            Diff string
        """
        original_lines = original.split('\n')
        edited_lines = edited.split('\n')
        
        diff_lines = []
        diff_lines.append("--- original")
        diff_lines.append("+++ edited")
        
        # Simple line-by-line diff
        max_lines = max(len(original_lines), len(edited_lines))
        
        for i in range(max_lines):
            orig_line = original_lines[i] if i < len(original_lines) else ""
            edit_line = edited_lines[i] if i < len(edited_lines) else ""
            
            if orig_line != edit_line:
                if orig_line:
                    diff_lines.append(f"- {orig_line}")
                if edit_line:
                    diff_lines.append(f"+ {edit_line}")
            else:
                diff_lines.append(f"  {orig_line}")
        
        return "\n".join(diff_lines)


class QuestionGenerator:
    """Generate clarifying questions for incomplete requests."""
    
    def __init__(self):
        self.question_templates = {
            "missing_details": [
                "Could you provide more details about {topic}?",
                "What specific {aspect} are you looking for?",
                "Can you clarify what you mean by {term}?"
            ],
            "ambiguous": [
                "Did you mean {option1} or {option2}?",
                "Are you referring to {context}?",
                "Which {category} would you prefer?"
            ],
            "incomplete": [
                "What should happen when {condition}?",
                "How should {component} behave?",
                "What's the expected {output}?"
            ]
        }
    
    def generate_questions(self,
                          request: str,
                          missing_info: List[str] = None) -> List[str]:
        """Generate clarifying questions.
        
        Args:
            request: User request
            missing_info: List of missing information types
            
        Returns:
            List of clarifying questions
        """
        questions = []
        
        # Detect missing information
        if not missing_info:
            missing_info = self._detect_missing_info(request)
        
        # Generate questions based on missing info
        for info_type in missing_info:
            if info_type in self.question_templates:
                template = self.question_templates[info_type][0]
                # Simple placeholder filling
                question = template.format(
                    topic="that",
                    aspect="aspect",
                    term="that",
                    option1="option A",
                    option2="option B",
                    context="this context",
                    category="type",
                    condition="this happens",
                    component="this",
                    output="result"
                )
                questions.append(question)
        
        return questions
    
    def _detect_missing_info(self, request: str) -> List[str]:
        """Detect what information is missing from request."""
        missing = []
        
        request_lower = request.lower()
        
        # Check for vague terms
        vague_terms = ["something", "thing", "stuff", "it", "that"]
        if any(term in request_lower for term in vague_terms):
            missing.append("missing_details")
        
        # Check for questions
        if "?" in request:
            missing.append("ambiguous")
        
        # Check for incomplete sentences
        if len(request.split()) < 5:
            missing.append("incomplete")
        
        return missing if missing else ["missing_details"]


class ContentGenerator:
    """Main content generation system."""
    
    def __init__(self):
        self.code_generator = CodeGenerator()
        self.document_writer = DocumentWriter()
        self.email_composer = EmailComposer()
        self.editor = ContextAwareEditor()
        self.question_generator = QuestionGenerator()
        
        logging.info("Content Generator initialized")
    
    def generate(self,
                content_type: str,
                **kwargs) -> str:
        """Generate content of specified type.
        
        Args:
            content_type: Type of content (code, document, email)
            **kwargs: Content parameters
            
        Returns:
            Generated content
        """
        if content_type == "code":
            return self.code_generator.generate_code(**kwargs)
        
        elif content_type == "document":
            return self.document_writer.write_document(**kwargs)
        
        elif content_type == "email":
            return self.email_composer.compose_email(**kwargs)
        
        else:
            return f"Content type '{content_type}' not supported"
    
    def edit(self,
            original: str,
            instruction: str,
            context: Dict[str, Any] = None) -> Tuple[str, str]:
        """Edit content with context awareness."""
        return self.editor.edit_text(original, instruction, context)
    
    def clarify(self, request: str) -> List[str]:
        """Generate clarifying questions for request."""
        return self.question_generator.generate_questions(request)


# Global instance
_content_generator: Optional[ContentGenerator] = None


def get_content_generator() -> ContentGenerator:
    """Get global content generator instance."""
    global _content_generator
    
    if _content_generator is None:
        _content_generator = ContentGenerator()
    
    return _content_generator

"""
AI-related functions for transcript analysis with enhanced precision.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
import logging
import re
import json

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models import Model
from pydantic import BaseModel, Field, ValidationError
from httpx import AsyncClient

from .config import Config

logger = logging.getLogger(__name__)
config = Config()

class TranscriptSegment(BaseModel):
    """Represents a relevant segment of transcript with precise timing."""
    start_time: str = Field(description="Start timestamp in MM:SS format")
    end_time: str = Field(description="End timestamp in MM:SS format")
    text: str = Field(description="The transcript text for this segment")
    relevance_score: float = Field(description="Relevance score from 0.0 to 1.0", ge=0.0, le=1.0)
    reasoning: str = Field(default="", description="Explanation for why this segment is relevant")

class TranscriptAnalysis(BaseModel):
    """Analysis result for transcript segments."""
    most_relevant_segments: List[TranscriptSegment]
    summary: str = Field(description="Brief summary of the video content")
    key_topics: List[str] = Field(description="List of main topics discussed")
    reasoning: str = Field(default="", description="AI reasoning/thinking for the analysis")

# Simplified system prompt that trusts AssemblyAI timing
simplified_system_prompt = """You are an expert at analyzing video transcripts to find the most engaging segments for short-form content creation.

CORE OBJECTIVES:
1. Identify segments that would be compelling on social media platforms
2. Focus on complete thoughts, insights, or entertaining moments
3. Prioritize content with hooks, emotional moments, or valuable information
4. Each segment should be engaging and worth watching

SEGMENT SELECTION CRITERIA:
1. STRONG HOOKS: Attention-grabbing opening lines
2. VALUABLE CONTENT: Tips, insights, interesting facts, stories
3. EMOTIONAL MOMENTS: Excitement, surprise, humor, inspiration
4. COMPLETE THOUGHTS: Self-contained ideas that make sense alone
5. ENTERTAINING: Content people would want to share

TIMING GUIDELINES:
- Segments MUST be between 10-45 seconds for optimal engagement
- CRITICAL: start_time MUST be different from end_time (minimum 10 seconds apart)
- Focus on natural content boundaries rather than arbitrary time limits
- Include enough context for the segment to be understandable

TIMESTAMP REQUIREMENTS - EXTREMELY IMPORTANT:
- Use EXACT timestamps as they appear in the transcript
- Never modify timestamp format (keep MM:SS structure)
- start_time MUST be LESS THAN end_time (start_time < end_time)
- MINIMUM segment duration: 10 seconds (end_time - start_time >= 10 seconds)
- Look at transcript ranges like [02:25 - 02:35] and use different start/end times
- NEVER use the same timestamp for both start_time and end_time
- Example: start_time: "02:25", end_time: "02:35" (NOT "02:25" and "02:25")

IMPORTANT: You MUST return your final analysis as valid JSON in your message content. The JSON should have this structure:
{
  "most_relevant_segments": [
    {
      "start_time": "MM:SS",
      "end_time": "MM:SS",
      "text": "segment transcript",
      "relevance_score": 0.0-1.0,
      "reasoning": "why this segment is relevant"
    }
  ],
  "summary": "brief summary",
  "key_topics": ["topic1", "topic2"]
}

Find 3-7 compelling segments that would work well as standalone clips. Quality over quantity - choose segments that would genuinely engage viewers and have proper time ranges."""

# Create model based on config - allow overrides
def get_model(llm_config: Optional[Dict[str, Any]] = None):
    """
    Create an AI model with optional user-specific configuration.

    Args:
        llm_config: Dictionary with 'provider', 'model', 'api_key', 'base_url'
                   If not provided, uses environment config
    """
    if llm_config:
        # Use user's AI configuration
        provider = llm_config.get('provider', 'local')
        model_id = llm_config.get('model', 'llama3')
        api_key = llm_config.get('api_key')
        base_url = llm_config.get('base_url')

        if provider == 'local' or provider == 'openai':
            openai_provider = OpenAIProvider(
                base_url=base_url or config.llm_base_url,
                api_key=api_key or config.openai_api_key or "local"
            )
            return OpenAIModel(model_id, provider=openai_provider)

        # For future support of other providers
        if provider == 'google' or provider == 'anthropic':
            # Use default OpenAI-compatible config for now
            openai_provider = OpenAIProvider(
                base_url=config.llm_base_url,
                api_key=api_key or config.openai_api_key
            )
            return OpenAIModel(model_id, provider=openai_provider)

    # Fall back to environment config
    model_name = config.llm
    if model_name.startswith('openai:') or config.llm_base_url:
        model_id = model_name.split(':', 1)[1] if ':' in model_name else model_name
        provider = OpenAIProvider(
            base_url=config.llm_base_url,
            api_key=config.openai_api_key or "local"
        )
        return OpenAIModel(model_id, provider=provider)

    return model_name

async def call_model_with_reasoning(model: Model, system_prompt: str, user_prompt: str) -> tuple[TranscriptAnalysis, str]:
    """
    Call the model and extract both the structured output and reasoning.

    Returns:
        tuple: (TranscriptAnalysis, reasoning_text)
    """
    reasoning = ""

    # First, try using the standard pydantic-ai agent
    agent = Agent(
        model=model,
        result_type=TranscriptAnalysis,
        system_prompt=system_prompt
    )

    try:
        result = await agent.run(user_prompt)
        analysis = result.data

        # Try to extract reasoning from messages
        try:
            all_messages = result.all_messages()
            for msg in reversed(all_messages):
                if hasattr(msg, 'content'):
                    # Handle different message content types
                    content = msg.content
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and 'reasoning' in part:
                                reasoning = part['reasoning']
                                logger.info(f"Captured reasoning from reasoning field ({len(reasoning)} chars)")
                                break
                    elif isinstance(content, dict):
                        if 'reasoning' in content:
                            reasoning = content['reasoning']
                            logger.info(f"Captured reasoning from reasoning field ({len(reasoning)} chars)")
                        elif 'choices' in content and content['choices']:
                            # Handle OpenAI API response format
                            choice = content['choices'][0]
                            if isinstance(choice, dict) and 'message' in choice:
                                message = choice['message']
                                if 'reasoning' in message:
                                    reasoning = message['reasoning']
                                    logger.info(f"Captured reasoning from message.reasoning ({len(reasoning)} chars)")
                    elif isinstance(content, str):
                        # Check for thinking tags in string content
                        if '<thinking>' in content or '' in content:
                            thinking_match = re.search(r'<thinking>(.*?)</thinking>', content, re.DOTALL) or \
                                           re.search(r'(.*?)', content, re.DOTALL)
                            if thinking_match:
                                reasoning = thinking_match.group(1).strip()
                                logger.info(f"Captured reasoning from thinking tags ({len(reasoning)} chars)")
                if reasoning:
                    break
        except Exception as e:
            logger.warning(f"Could not extract reasoning: {e}")

        # Add reasoning to analysis if we found it
        if reasoning and not analysis.reasoning:
            # Create new analysis with reasoning
            analysis = TranscriptAnalysis(
                most_relevant_segments=analysis.most_relevant_segments,
                summary=analysis.summary,
                key_topics=analysis.key_topics,
                reasoning=reasoning
            )

        return analysis, reasoning

    except Exception as e:
        # If pydantic-ai fails (e.g., model puts JSON in reasoning field),
        # try manual parsing from the raw API response
        logger.warning(f"Standard agent failed, trying manual parsing: {e}")

        # Make a direct API call to get the raw response
        try:
            if isinstance(model, OpenAIModel):
                provider = model._provider
                client = AsyncClient()
                response = await client.post(
                    f"{provider.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {provider.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model.name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.1,
                    },
                    timeout=120.0
                )

                response_data = response.json()

                # Extract reasoning from the response
                if 'choices' in response_data and response_data['choices']:
                    choice = response_data['choices'][0]
                    message = choice.get('message', {})

                    # Check for reasoning field
                    reasoning = message.get('reasoning', '')

                    # Get content
                    content = message.get('content', '')

                    # If content is empty but reasoning exists, try to parse JSON from reasoning
                    if not content and reasoning:
                        logger.info(f"Content empty, trying to parse JSON from reasoning field")
                        content = reasoning
                    elif reasoning and content:
                        # Both exist, use content for parsing
                        pass

                    # Try to extract JSON from content
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        json_str = json_match.group(0)
                        data = json.loads(json_str)

                        # Parse the TranscriptAnalysis from the JSON
                        segments_data = data.get('most_relevant_segments', [])
                        segments = []
                        for seg_data in segments_data:
                            segment = TranscriptSegment(
                                start_time=seg_data.get('start_time', ''),
                                end_time=seg_data.get('end_time', ''),
                                text=seg_data.get('text', ''),
                                relevance_score=seg_data.get('relevance_score', 0.5),
                                reasoning=seg_data.get('reasoning', '')
                            )
                            segments.append(segment)

                        analysis = TranscriptAnalysis(
                            most_relevant_segments=segments,
                            summary=data.get('summary', ''),
                            key_topics=data.get('key_topics', []),
                            reasoning=reasoning
                        )

                        logger.info(f"Successfully parsed analysis from reasoning field")
                        return analysis, reasoning

        except Exception as parse_error:
            logger.error(f"Manual parsing also failed: {parse_error}")

        # Re-raise the original error if we couldn't parse
        raise

# Default agent using environment config (kept for compatibility)
transcript_agent = None

async def get_most_relevant_parts_by_transcript(
    transcript: str,
    llm_config: Optional[Dict[str, Any]] = None
) -> TranscriptAnalysis:
    """Get the most relevant parts of a transcript for creating clips - simplified version."""
    logger.info(f"Starting AI analysis of transcript ({len(transcript)} chars)")

    try:
        model = get_model(llm_config)

        analysis, reasoning = await call_model_with_reasoning(
            model=model,
            system_prompt=simplified_system_prompt,
            user_prompt=f"""Analyze this video transcript and identify the most engaging segments for short-form content.

Find segments that would be compelling as standalone clips for social media.

Transcript:
{transcript}"""
        )

        logger.info(f"AI analysis found {len(analysis.most_relevant_segments)} segments")

        # Simple validation - just ensure segments have content
        validated_segments = []
        for segment in analysis.most_relevant_segments:
            # Handle text that might be a list (AI sometimes returns lists)
            text_content = segment.text
            if isinstance(text_content, list):
                text_content = ' '.join(str(item) for item in text_content)
                logger.warning(f"Segment text was a list, converted to string")
            text_content = str(text_content)

            # Validate text content
            if not text_content.strip() or len(text_content.split()) < 3:  # At least 3 words
                logger.warning(f"Skipping segment with insufficient content: '{text_content[:50]}...'")
                continue

            # Normalize timestamps that might be lists
            start_time = str(segment.start_time[0]) if isinstance(segment.start_time, list) else str(segment.start_time)
            end_time = str(segment.end_time[0]) if isinstance(segment.end_time, list) else str(segment.end_time)

            # Validate timestamps - CRITICAL: start and end must be different
            if start_time == end_time:
                logger.warning(f"Skipping segment with identical start/end times: {start_time}")
                continue

            # Parse timestamps to validate duration
            try:
                start_parts = start_time.split(':')
                end_parts = end_time.split(':')

                start_seconds = int(start_parts[0]) * 60 + int(start_parts[1])
                end_seconds = int(end_parts[0]) * 60 + int(end_parts[1])

                duration = end_seconds - start_seconds

                if duration <= 0:
                    logger.warning(f"Skipping segment with invalid duration: {start_time} to {end_time} = {duration}s")
                    continue

                if duration < 5:  # Minimum 5 seconds
                    logger.warning(f"Skipping segment too short: {duration}s (min 5s required)")
                    continue

                # Update segment with normalized values
                segment.text = text_content
                validated_segments.append(segment)
                logger.info(f"Validated segment: {start_time}-{end_time} ({duration}s)")

            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping segment with invalid timestamp format: {segment.start_time}-{segment.end_time}: {e}")
                continue

        # Sort by relevance
        validated_segments.sort(key=lambda x: x.relevance_score, reverse=True)

        final_analysis = TranscriptAnalysis(
            most_relevant_segments=validated_segments,
            summary=analysis.summary,
            key_topics=analysis.key_topics,
            reasoning=reasoning
        )

        logger.info(f"Selected {len(validated_segments)} segments for processing")
        if validated_segments:
            logger.info(f"Top segment score: {validated_segments[0].relevance_score:.2f}")

        return final_analysis

    except Exception as e:
        logger.error(f"Error in transcript analysis: {e}")
        return TranscriptAnalysis(
            most_relevant_segments=[],
            summary=f"Analysis failed: {str(e)}",
            key_topics=[],
            reasoning=""
        )
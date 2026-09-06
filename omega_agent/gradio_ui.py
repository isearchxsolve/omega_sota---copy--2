"""Gradio stateful chat interface for OMEGA Agent."""

import asyncio
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr

from omega_agent import OmegaAgent, Config
from omega_agent.core.types import AgentResult


class UIState(Enum):
    """UI state machine."""
    IDLE = "idle"
    RUNNING = "running"
    WAITING_FOR_INPUT = "waiting_for_input"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ConversationState:
    """State for a single conversation session."""
    messages: List[Tuple[str, str]] = field(default_factory=list)
    current_state: UIState = UIState.IDLE
    pending_request: Optional[Dict[str, Any]] = None
    current_goal: Optional[str] = None
    agent_result: Optional[AgentResult] = None
    execution_context: Dict[str, Any] = field(default_factory=dict)


class OmegaAgentUI:
    """
    Wrapper for OmegaAgent with UI callback support.
    
    Detects when agent needs credentials/details and pauses execution
    to request user input via callback.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.agent = OmegaAgent(config=self.config)
        self.input_callback: Optional[Callable[[str, Dict[str, Any]], str]] = None
        self.pending_input: Optional[str] = None
        self._input_event = threading.Event()

    def set_input_callback(self, callback: Callable[[str, Dict[str, Any]], str]):
        """Set callback for requesting user input during execution."""
        self.input_callback = callback

    async def run_with_interactive_input(
        self,
        goal: str,
        domain: Optional[str] = None,
        max_time: Optional[int] = None,
    ) -> Tuple[AgentResult, Optional[Dict[str, Any]]]:
        """
        Run agent with interactive input support.
        
        Returns:
            Tuple of (result, input_request) where input_request is None
            if execution completed without needing input, or contains
            the request details if input is needed.
        """
        # Check if output contains credential/detail requests
        result = await self.agent.run(goal=goal, domain=domain, max_time=max_time)
        
        # Parse result for credential/detail requests
        input_request = self._parse_input_request(result)
        
        return result, input_request

    def _parse_input_request(self, result: AgentResult) -> Optional[Dict[str, Any]]:
        """
        Parse agent result for credential/detail requests.
        
        Looks for patterns like:
        - "Please provide your API key"
        - "I need your credentials for..."
        - "Please specify: ..."
        """
        if not result or not result.output:
            return None
            
        output = result.output.lower()
        
        # Credential request patterns
        credential_patterns = [
            r"(please provide|need|require).*(api key|api_key|apikey|token|password|credential)",
            r"(please provide|need|require).*(access key|access_key|accesskey)",
            r"(please provide|need|require).*(secret key|secret_key|secretkey)",
        ]
        
        # Detail request patterns
        detail_patterns = [
            r"please specify",
            r"please provide more details?",
            r"i need more information about",
            r"clarif",
        ]
        
        for pattern in credential_patterns + detail_patterns:
            if re.search(pattern, output):
                # Extract the request type
                request_type = "credential" if "key" in pattern or "token" in pattern or "password" in pattern else "detail"
                
                return {
                    "type": request_type,
                    "message": self._extract_request_message(result.output),
                    "context": {
                        "domain": result.domain,
                        "action": result.decision.action if result.decision else None,
                    }
                }
        
        return None

    def _extract_request_message(self, output: str) -> str:
        """Extract the specific request message from output."""
        # Find sentences that contain request keywords
        sentences = output.split(". ")
        request_sentences = []
        
        keywords = ["please", "need", "require", "provide", "specify"]
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                request_sentences.append(sentence.strip())
        
        if request_sentences:
            return ". ".join(request_sentences[:2])  # Return first 2 relevant sentences
        
        return output[:200]  # Fallback to first 200 chars


class GradioInterface:
    """Gradio ChatInterface for OMEGA Agent with stateful conversation."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.agent_ui = OmegaAgentUI(config=self.config)
        self.conversation_state = ConversationState()
        self._lock = threading.Lock()

    def _format_agent_response(self, result: AgentResult) -> str:
        """Format agent result for chat display."""
        if not result.success:
            return f"❌ **Error:** {result.output}"
        
        response = f"✅ **Success**\n\n"
        response += f"**Domain:** {result.domain}\n"
        response += f"**Route:** {result.route}\n"
        
        if result.decision:
            response += f"**Action:** {result.decision.action}\n"
            response += f"**Confidence:** {result.decision.confidence:.0%}\n\n"
        
        response += f"**Output:**\n{result.output}\n\n"
        response += f"**Cost:** ${result.cost:.4f} | **Latency:** {result.latency:.2f}s"
        
        return response

    def _format_input_request(self, request: Dict[str, Any]) -> str:
        """Format input request for user."""
        emoji = "🔑" if request["type"] == "credential" else "❓"
        return f"{emoji} **{request['type'].title()} Required**\n\n{request['message']}\n\nPlease provide the requested information above."

    async def _run_goal_async(
        self,
        goal: str,
        domain: Optional[str] = None,
        max_time: Optional[int] = None,
    ) -> str:
        """Async wrapper for running agent goal."""
        try:
            result, input_request = await self.agent_ui.run_with_interactive_input(
                goal=goal,
                domain=domain,
                max_time=max_time,
            )
            
            with self._lock:
                self.conversation_state.agent_result = result
                self.conversation_state.current_state = UIState.COMPLETE
            
            if input_request:
                with self._lock:
                    self.conversation_state.pending_request = input_request
                    self.conversation_state.current_state = UIState.WAITING_FOR_INPUT
                
                return self._format_input_request(input_request)
            
            return self._format_agent_response(result)
            
        except Exception as e:
            with self._lock:
                self.conversation_state.current_state = UIState.ERROR
            
            return f"❌ **Execution Error:** {str(e)}"

    def handle_message(
        self,
        message: str,
        history: List[Tuple[str, str]],
        domain: Optional[str] = None,
        max_time: Optional[int] = None,
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Handle user message in chat interface.
        
        This is the main entry point for Gradio ChatInterface.
        """
        with self._lock:
            # Check if we're waiting for input from a previous request
            if self.conversation_state.current_state == UIState.WAITING_FOR_INPUT:
                # User provided the requested input
                # For now, we'll acknowledge and re-run with the input
                # In a full implementation, this would inject the input into context
                response = f"✅ **Input Received:** {message}\n\nRe-running goal with provided input..."
                
                # Reset state and re-run
                self.conversation_state.current_state = UIState.RUNNING
                self.conversation_state.pending_request = None
                
                # Re-run the original goal
                goal = self.conversation_state.current_goal or message
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self._run_goal_async(goal, domain, max_time)
                    )
                finally:
                    loop.close()
                
                # Update history
                history.append((message, result))
                return result, history
            
            # New goal
            self.conversation_state.current_goal = message
            self.conversation_state.current_state = UIState.RUNNING
            self.conversation_state.messages = history
        
        # Run the agent
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                self._run_goal_async(message, domain, max_time)
            )
        finally:
            loop.close()
        
        # Update history
        with self._lock:
            history.append((message, response))
            self.conversation_state.messages = history
            self.conversation_state.current_state = UIState.IDLE
        
        return response, history

    def clear_conversation(self) -> Tuple[List[Tuple[str, str]], str]:
        """Clear conversation state."""
        with self._lock:
            self.conversation_state = ConversationState()
        return [], "Conversation cleared. Ready for new goals."

    def get_state_info(self) -> str:
        """Get current state information for display."""
        with self._lock:
            state = self.conversation_state.current_state
            return f"**State:** {state.value}\n**Messages:** {len(self.conversation_state.messages)}"

    def create_interface(self) -> gr.Blocks:
        """Create and return the Gradio interface."""
        with gr.Blocks(title="OMEGA Agent - Stateful Chat Interface") as interface:
            gr.Markdown("# 🤖 OMEGA Agent - Stateful Chat Interface")
            gr.Markdown("Interact with OMEGA to execute goals. OMEGA will request credentials/details when needed.")
            
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=500,
                        show_copy_button=True,
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Your Goal",
                            placeholder="Enter your goal here...",
                            lines=2,
                            scale=4,
                        )
                        submit_btn = gr.Button("Submit", variant="primary", scale=1)
                    
                    with gr.Accordion("Advanced Options", open=False):
                        domain_input = gr.Textbox(
                            label="Domain (optional)",
                            placeholder="e.g., crypto_trading, research, coding",
                        )
                        max_time_input = gr.Number(
                            label="Max Time (seconds)",
                            value=300,
                            minimum=10,
                            maximum=600,
                        )
                
                with gr.Column(scale=1):
                    state_display = gr.Markdown("**State:** idle")
                    clear_btn = gr.Button("Clear Conversation", variant="secondary")
                    
                    gr.Markdown("### Agent Info")
                    info_display = gr.Markdown(self._get_agent_info())
            
            # Event handlers
            submit_btn.click(
                fn=self.handle_message,
                inputs=[msg_input, chatbot, domain_input, max_time_input],
                outputs=[msg_input, chatbot],
            )
            
            msg_input.submit(
                fn=self.handle_message,
                inputs=[msg_input, chatbot, domain_input, max_time_input],
                outputs=[msg_input, chatbot],
            )
            
            clear_btn.click(
                fn=self.clear_conversation,
                outputs=[chatbot, state_display],
            )
            
            # Update state display on interaction
            chatbot.change(
                fn=self.get_state_info,
                outputs=[state_display],
            )
        
        return interface

    def _get_agent_info(self) -> str:
        """Get agent configuration info."""
        provider = self.config.active_llm_provider() or "mock"
        return f"""
**LLM Provider:** {provider}
**Primary Model:** {self.config.primary_model}
**Domain Focus:** {self.config.domain_focus or 'auto-detect'}
**Dynamic Discovery:** {'✅' if self.config.enable_dynamic_discovery else '❌'}
**Learning Enabled:** {'✅' if self.config.enable_learning else '❌'}
"""

    def launch(
        self,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False,
        debug: bool = False,
    ):
        """Launch the Gradio interface."""
        interface = self.create_interface()
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            debug=debug,
        )


def create_gradio_ui(config: Optional[Config] = None) -> GradioInterface:
    """Factory function to create Gradio interface."""
    return GradioInterface(config=config)


if __name__ == "__main__":
    # Launch the UI when run directly
    ui = create_gradio_ui()
    ui.launch(share=True)

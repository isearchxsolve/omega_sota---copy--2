"""OMEGA CLI — run goals from the command line."""

import argparse
import asyncio
import json
import sys

from omega_agent import OmegaAgent, Config


def main():
    parser = argparse.ArgumentParser(
        description="OMEGA Agent — True Master of All Trades, True Action Taker",
    )
    parser.add_argument("goal", nargs="?", help="Goal to execute")
    parser.add_argument("--domain", "-d", help="Domain hint (crypto_trading, research, coding, planning)")
    parser.add_argument("--max-time", "-t", type=int, default=300, help="Max execution time (seconds)")
    parser.add_argument("--mock", action="store_true", help="Force mock LLM mode (ignore GROQ_API_KEY)")
    parser.add_argument("--json", action="store_true", help="Output JSON result")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    parser.add_argument("--port", type=int, default=8000, help="API port (with --serve)")
    parser.add_argument("--benchmark", action="store_true", help="Run SOTA benchmark suite")
    parser.add_argument("--gradio", action="store_true", help="Start Gradio chat UI")
    parser.add_argument("--gradio-port", type=int, default=7860, help="Gradio UI port (with --gradio)")
    parser.add_argument("--share", action="store_true", help="Create public share link (with --gradio)")

    args = parser.parse_args()

    if args.serve:
        import uvicorn
        uvicorn.run("omega_agent.api:app", host="0.0.0.0", port=args.port, reload=False)
        return

    if args.benchmark:
        from test_omega_sota import run_full_benchmark
        report = asyncio.run(run_full_benchmark())
        print(json.dumps({
            "winner": report.winner,
            "test_cases": report.test_cases_count,
            "baselines": report.baselines,
        }, indent=2))
        return

    if args.gradio:
        from omega_agent.gradio_ui import create_gradio_ui
        config = Config(use_mock_llm=args.mock)
        ui = create_gradio_ui(config)
        print(f"🚀 Launching OMEGA Gradio UI on port {args.gradio_port}...")
        ui.launch(
            server_name="0.0.0.0",
            server_port=args.gradio_port,
            share=args.share,
            debug=False,
        )
        return

    if not args.goal:
        parser.print_help()
        sys.exit(1)

    config = Config(use_mock_llm=args.mock)
    agent = OmegaAgent(config=config)
    result = asyncio.run(agent.run(goal=args.goal, domain=args.domain, max_time=args.max_time))

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\n{'='*60}")
        print("OMEGA RESULT")
        print(f"{'='*60}")
        print(f"Success:  {result.success}")
        print(f"Domain:   {result.domain}")
        print(f"Action:   {result.decision.action if result.decision else 'N/A'}")
        print(f"Confidence: {result.decision.confidence:.0%}" if result.decision else "")
        print(f"Latency:  {result.latency:.2f}s")
        print(f"Cost:     ${result.cost:.4f}")
        print(f"\n{result.output}")
        print(f"{'='*60}\n")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()

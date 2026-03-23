"""
main.py — CLI entry point for the Blog Writing Agent.
Can be used independently of the Streamlit UI for scripting / testing.

Usage:
    python main.py --topic "Future of AI" --keywords "AI, LLM" --length medium --images 3
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


def main():
    parser = argparse.ArgumentParser(description="AI Blog Writing Agent CLI")
    parser.add_argument("--topic",        required=True,  help="Blog topic")
    parser.add_argument("--keywords",     default="",     help="Comma-separated keywords")
    parser.add_argument("--tone",         default="informative and engaging")
    parser.add_argument("--length",       default="medium", choices=["short", "medium", "long"])
    parser.add_argument("--images",       default=3, type=int, help="Number of images (0-6)")
    parser.add_argument("--language",     default="English")
    parser.add_argument("--instructions", default="", help="Custom instructions")
    args = parser.parse_args()

    from graph import run_pipeline

    logger.info("Starting Blog Writing Agent...")
    result = run_pipeline({
        "topic":               args.topic,
        "keywords":            args.keywords,
        "tone":                args.tone,
        "length":              args.length,
        "num_images":          args.images,
        "language":            args.language,
        "custom_instructions": args.instructions,
    })

    print("\n" + "=" * 60)
    if result.error:
        print(f"❌ FAILED: {result.error}")
        sys.exit(1)
    else:
        print(f"✅ Blog generated: {result.outline.title}")
        print(f"   Words: {len(result.final_markdown.split())}")
        print(f"   Images: {sum(1 for s in result.outline.sections if s.image_path)}")
        print("=" * 60)


if __name__ == "__main__":
    main()

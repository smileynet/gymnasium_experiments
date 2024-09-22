import logging
import os

from submission_helper import SubmissionHelper


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def main():
    setup_logging()
    helper = SubmissionHelper()

    try:
        helper.load_model()
    except FileNotFoundError as e:
        logging.error(f"Model file not found: {e}")
        return

    try:
        helper.evaluate_and_record()
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        return

    if helper.prompt_for_submission():
        try:
            if not helper.prepare_submission_files():
                logging.error("Failed to prepare submission files")
                return

            if not helper.validate_submission_files():
                logging.error("Submission files validation failed")
                return

            helper.submit_to_hub()
        except Exception as e:
            logging.error(f"Error during submission process: {e}")
        finally:
            helper.cleanup()
    else:
        logging.info("Model submission cancelled")


if __name__ == "__main__":
    main()

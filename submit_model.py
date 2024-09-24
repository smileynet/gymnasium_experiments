from logging_config import console, logger
from submission_helper import SubmissionHelper


def main():
    logger.info("Starting submission process")
    helper = SubmissionHelper()

    try:
        helper.load_model()
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        return

    try:
        helper.evaluate_and_record()
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return

    if helper.prompt_for_submission():
        try:
            if not helper.prepare_submission_files():
                logger.error("Failed to prepare submission files")
                return

            if not helper.validate_submission_files():
                logger.error("Submission files validation failed")
                return

            console.print("Submitting to the hub...", style="bright_green")
            helper.submit_to_hub()
        except Exception as e:
            logger.error(f"Error during submission process: {e}")
        finally:
            helper.cleanup()
    else:
        console.print("Model submission cancelled", style="bright_red")


if __name__ == "__main__":
    main()

from logging_config import console, logger
from submission_helper import SubmissionHelper


def main():
    """
    Main function to handle the model submission process.

    This function orchestrates the entire submission process, including:
    1. Loading the model
    2. Evaluating and recording results
    3. Preparing submission files
    4. Validating submission files
    5. Submitting to the hub (if confirmed by the user)

    The function uses a SubmissionHelper instance to perform these tasks
    and handles various exceptions that may occur during the process.
    """
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

    try:
        if not helper.prepare_submission_files():
            logger.error("Failed to prepare submission files")
            return

        if not helper.validate_submission_files():
            logger.error("Submission files validation failed")
            return

        if helper.prompt_for_submission():
            console.print("Submitting to the hub...", style="bright_green")
            helper.submit_to_hub()
        else:
            console.print("Model submission cancelled", style="bright_red")
    except Exception as e:
        logger.error(f"Error during submission process: {e}")
    finally:
        helper.cleanup()


if __name__ == "__main__":
    main()

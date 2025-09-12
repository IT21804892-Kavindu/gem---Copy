from playwright.sync_api import sync_playwright, Page, expect

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    try:
        # 1. Go to the application page.
        page.goto("http://localhost:5173/")

        # Take a screenshot immediately to see the initial page state.
        page.screenshot(path="jules-scratch/verification/debug_screenshot.png")
        print("Debug screenshot taken.")

        # 2. Fill out the sensor data form.
        page.get_by_label("Average Rainfall (mm)").fill("50")
        page.get_by_label("Temperature (°C)").fill("30")
        page.get_by_label("Water Content (0-100)").fill("60")
        page.get_by_label("7-day Average Rainfall (mm)").fill("45")
        page.get_by_label("7-day Average Water Content (0-100)").fill("55")

        # 3. Click the submit button.
        page.get_by_role("button", name="Generate Prediction").click()

        # 4. Wait for the prediction display to update.
        prediction_heading = page.get_by_role("heading", name="Prediction Results")
        expect(prediction_heading).to_be_visible(timeout=15000)

        loading_spinner = page.get_by_text("Calculating...")
        expect(loading_spinner).not_to_be_visible()

        # 5. Take a screenshot of the page.
        page.screenshot(path="jules-scratch/verification/prediction_success.png")
        print("Success screenshot taken.")

    except Exception as e:
        print(f"An error occurred: {e}")
        # Take a final screenshot on error to see the state at the time of failure.
        page.screenshot(path="jules-scratch/verification/error.png")

    finally:
        # 6. Clean up.
        context.close()
        browser.close()

with sync_playwright() as playwright:
    run(playwright)

/** The bug report dialog. */

import { api } from "./api.js";
import { element, errorMessage, formString, notice } from "./dom.js";
import { currentGame } from "./state.js";
import { BugReportReceipt } from "./types.js";

export function openBugReport(): void {
  const form = element<HTMLFormElement>("#bug-report-form");
  form.reset();
  element("#bug-report-error").textContent = "";
  const dialog = element<HTMLDialogElement>("#bug-report-dialog");
  dialog.showModal();
  element<HTMLTextAreaElement>(
    '#bug-report-form textarea[name="expectedBehavior"]',
  ).focus();
}

export function closeBugReport(): void {
  element<HTMLDialogElement>("#bug-report-dialog").close();
}

export async function submitBugReport(form: HTMLFormElement): Promise<void> {
  const submit = form.querySelector<HTMLButtonElement>('button[type="submit"]');
  if (!submit) throw new Error("Bug report submit button is missing");
  submit.disabled = true;
  element("#bug-report-error").textContent = "";
  try {
    const current = currentGame();
    const fields = new FormData(form);
    const report = await api<BugReportReceipt>(
      `/api/games/${current.summary.id}/bugs`,
      {
        method: "POST",
        body: {
          version: current.summary.version,
          expectedBehavior: formString(fields, "expectedBehavior"),
          actualBehavior: formString(fields, "actualBehavior"),
        },
      },
    );
    closeBugReport();
    notice(`Bug report #${report.id} submitted. Thank you.`);
  } catch (error) {
    element("#bug-report-error").textContent = errorMessage(error);
  } finally {
    submit.disabled = false;
  }
}

/** The Help dialog: board colours and the rules. */

import { element, errorMessage, escapeHtml } from "./dom.js";

export let rulesLoaded = false;

export async function openHelp(): Promise<void> {
  element<HTMLDialogElement>("#help-dialog").showModal();
  if (rulesLoaded) return;
  const rules = element("#rules");
  try {
    const response = await fetch("/assets/rules.md");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    rules.innerHTML = renderRules(await response.text());
    rulesLoaded = true;
  } catch (error) {
    rules.innerHTML = `<p class="help-note">The rules could not be loaded (${escapeHtml(errorMessage(error))}).</p>`;
  }
}

/**
 * Renders the subset of Markdown that RULES.md uses: headings, `* ` bullets,
 * paragraphs, bold and inline code. Headings drop one level to sit under the
 * dialog's own title.
 */
export function renderRules(markdown: string): string {
  const inline = (text: string): string =>
    escapeHtml(text)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`(.+?)`/g, "<code>$1</code>");
  const html: string[] = [];
  let paragraph: string[] = [];
  let list: string[] = [];
  const flush = (): void => {
    if (paragraph.length > 0)
      html.push(`<p>${inline(paragraph.join(" "))}</p>`);
    if (list.length > 0)
      html.push(
        `<ul>${list.map((item) => `<li>${inline(item)}</li>`).join("")}</ul>`,
      );
    paragraph = [];
    list = [];
  };
  for (const line of markdown.split("\n")) {
    const heading = /^(#{1,3})\s+(.*)$/.exec(line);
    const bullet = /^\*\s+(.*)$/.exec(line);
    if (heading?.[1] && heading[2] !== undefined) {
      flush();
      const level = heading[1].length + 2;
      html.push(`<h${level}>${inline(heading[2])}</h${level}>`);
    } else if (bullet?.[1] !== undefined) {
      if (paragraph.length > 0) flush();
      list.push(bullet[1]);
    } else if (line.trim() === "") {
      flush();
    } else if (list.length > 0 && /^\s/.test(line)) {
      list[list.length - 1] += ` ${line.trim()}`;
    } else {
      if (list.length > 0) flush();
      paragraph.push(line.trim());
    }
  }
  flush();
  return html.join("");
}

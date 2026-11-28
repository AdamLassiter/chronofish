/** Keeps the present line in view as it moves. */

export let presentScroll: { frame: number; cancel: () => void } | null = null;

/** Scrolls the board pane horizontally so the present line sits mid-view. */
export function followPresentLine(
  stage: HTMLElement,
  grid: HTMLElement,
  animate: boolean,
): void {
  presentScroll?.cancel();
  const marker = grid.querySelector<HTMLElement>(".present-line");
  if (!marker) return;
  const stageBox = stage.getBoundingClientRect();
  const markerBox = marker.getBoundingClientRect();
  const markerCentre =
    markerBox.left - stageBox.left + stage.scrollLeft + markerBox.width / 2;
  const maxLeft = stage.scrollWidth - stage.clientWidth;
  const target = Math.min(
    maxLeft,
    Math.max(0, markerCentre - stage.clientWidth / 2),
  );
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!animate || reducedMotion) {
    stage.scrollLeft = target;
    return;
  }
  const from = stage.scrollLeft;
  const distance = target - from;
  if (Math.abs(distance) < 1) return;
  const duration = 260;
  const started = performance.now();
  // Any manual scroll input hands control straight back to the user.
  const userEvents = ["wheel", "pointerdown", "touchstart", "keydown"] as const;
  const cancel = (): void => {
    if (!presentScroll) return;
    cancelAnimationFrame(presentScroll.frame);
    presentScroll = null;
    for (const name of userEvents) stage.removeEventListener(name, cancel);
  };
  for (const name of userEvents)
    stage.addEventListener(name, cancel, { passive: true });
  const step = (now: number): void => {
    const progress = Math.min(1, (now - started) / duration);
    const eased = 1 - (1 - progress) ** 3;
    stage.scrollLeft = from + distance * eased;
    if (progress < 1 && presentScroll) {
      presentScroll.frame = requestAnimationFrame(step);
    } else {
      cancel();
    }
  };
  presentScroll = { frame: requestAnimationFrame(step), cancel };
}

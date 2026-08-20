import { afterEach, vi } from "vitest";

// jsdom has no Web Audio and no getUserMedia, so the audio half of the client
// would be untestable without stand-ins. The stand-ins in tests/fakes.ts are
// installed per test; all that belongs here is what every test needs.

// The meter loop calls itself through requestAnimationFrame forever. Let the
// first tick run and stop there - none of these tests are about the meters.
vi.stubGlobal("requestAnimationFrame", () => 1);
vi.stubGlobal("cancelAnimationFrame", () => {});

// jsdom implements no scrolling of any kind, and the transcript panel scrolls
// itself to the newest line on every render.
Element.prototype.scrollTo = () => {};

afterEach(() => {
  sessionStorage.clear();
  vi.unstubAllEnvs();
});

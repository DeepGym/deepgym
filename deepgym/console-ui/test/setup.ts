import { GlobalRegistrator } from '@happy-dom/global-registrator';

// Component tests render into happy-dom. Register it before anything imports react-dom or motion,
// and tell React this is a test so it flushes work inside act().
GlobalRegistrator.register({ url: 'http://localhost/' });
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

// Finish animations at once. happy-dom also leaves a cancelled animation's rejection unhandled,
// which would fail whichever test is running.
const { MotionGlobalConfig } = await import('motion/react');
MotionGlobalConfig.skipAnimations = true;

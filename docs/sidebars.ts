import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/installation',
        'getting-started/configuration',
        'getting-started/quick-start',
      ],
    },
    {
      type: 'category',
      label: 'Core Concepts',
      items: [
        'concepts/architecture',
        'concepts/gpu-acceleration',
        'concepts/cross-mcp-workflows',
        'concepts/progressive-discovery',
      ],
    },
    {
      type: 'category',
      label: 'Visual Demos',
      link: {
        type: 'doc',
        id: 'demos/index',
      },
      items: [
        'demos/single-slit',
        'demos/double-slit',
        'demos/triple-slit',
        'demos/bragg-square',
        'demos/bragg-hexagonal',
        'demos/galaxy-collision',
      ],
    },
    {
      type: 'category',
      label: '1000+ Examples',
      link: {
        type: 'doc',
        id: 'examples/index',
      },
      items: [
        'examples/examples-physics',
        'examples/examples-chemistry',
        'examples/examples-mathematics',
        'examples/examples-ml-ai',
        'examples/examples-engineering',
        'examples/examples-biology',
        'examples/examples-finance',
        'examples/examples-data-science',
      ],
    },
  ],
  apiSidebar: [
    'api/overview',
    'api/math-mcp',
    'api/quantum-mcp',
    'api/molecular-mcp',
    'api/neural-mcp',
  ],
};

export default sidebars;

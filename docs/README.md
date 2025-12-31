# Math-Physics-ML MCP Documentation Site

This is a Docusaurus documentation site for the Math-Physics-ML MCP system.

## Current Status

### Completed:
- ✅ Docusaurus site initialized with TypeScript
- ✅ Custom configuration for Math-Physics-ML MCP branding
- ✅ Homepage/intro page with system overview
- ✅ Getting Started documentation:
  - Installation guide
  - Configuration guide (Claude Desktop & Claude Code)
  - Quick Start tutorial with examples for all 4 MCPs
- ✅ Architecture documentation
- ✅ API Overview page
- ✅ Math MCP API reference (complete)

### In Progress:
- API documentation for Quantum, Molecular, and Neural MCPs
- Additional concept pages (GPU acceleration, cross-MCP workflows, progressive discovery)
- Usage guides for specific scenarios
- Advanced topics

## Building the Documentation

### Development Server

Start the development server with hot reload:

```bash
cd /home/beagle/work/math-mcp/docs
pnpm start
```

The site will be available at http://localhost:3000

### Production Build

Build the static site:

```bash
cd /home/beagle/work/math-mcp/docs
pnpm build
```

**Note:** There are currently some MDX syntax issues with angle brackets and curly braces in code comments that need to be resolved for a successful build.

### Serve Production Build

After building, serve the static site locally:

```bash
cd /home/beagle/work/math-mcp/docs
pnpm serve
```

## Structure

```
docs/
├── docs/                    # Documentation content
│   ├── intro.md            # Homepage
│   ├── getting-started/    # Installation, configuration, quick start
│   ├── concepts/           # Architecture and core concepts
│   ├── guides/             # Usage guides (to be created)
│   ├── advanced/           # Advanced topics (to be created)
│   └── api/                # API reference
│       ├── overview.md     # API overview
│       ├── math-mcp.md     # Math MCP complete API
│       └── [other MCPs]    # To be created
├── src/                    # React components and custom pages
├── static/                 # Static assets
├── docusaurus.config.ts    # Site configuration
└── sidebars.ts            # Sidebar navigation
```

## Customization

### Site Metadata

Edit `docusaurus.config.ts` to customize:
- Site title and tagline
- URLs and deployment settings
- Navigation bar and footer links
- Theme colors and styling

### Sidebar Navigation

Edit `sidebars.ts` to add/remove pages from the sidebar navigation.

**Note:** All pages referenced in sidebars.ts must exist as .md files in the docs/ directory.

## Adding New Documentation

### Create a New Page

1. Create a new markdown file in `docs/docs/[category]/page-name.md`
2. Add frontmatter at the top:
   ```markdown
   ---
   sidebar_position: 1
   ---

   # Page Title

   Content here...
   ```
3. Add the page to `sidebars.ts` in the appropriate category
4. Restart the dev server or rebuild

### MDX Syntax Considerations

Docusaurus uses MDX (Markdown + JSX). Be careful with:
- **Curly braces `{}`** in code comments: Wrap in backticks or escape
- **Angle brackets `<>`**: Use "less than" / "greater than" in prose
- **JSX elements**: Can be embedded directly in markdown

Examples:
```markdown
# Good
Returns: `{result: "value"}`

# Bad (causes parsing errors)
Returns: {result: "value"}
```

## Deployment

### GitHub Pages

```bash
pnpm deploy
```

### Manual Deployment

1. Build the site: `pnpm build`
2. The static files will be in `build/`
3. Deploy the `build/` directory to any static hosting service

## Next Steps

To complete the documentation:

1. **Fix MDX syntax issues** in existing files (curly braces, angle brackets)
2. **Create API documentation** for:
   - Quantum MCP (api/quantum-mcp.md)
   - Molecular MCP (api/molecular-mcp.md)
   - Neural MCP (api/neural-mcp.md)
3. **Add concept pages**:
   - GPU acceleration deep dive
   - Cross-MCP workflows examples
   - Progressive discovery patterns
4. **Create usage guides**:
   - Math workflows
   - Quantum simulations
   - Molecular dynamics
   - Neural training
5. **Add advanced topics**:
   - Performance optimization
   - Custom potentials
   - Multi-MCP pipelines

## Resources

- [Docusaurus Documentation](https://docusaurus.io/docs)
- [MDX Documentation](https://mdxjs.com/)
- [Model Context Protocol](https://modelcontextprotocol.io)

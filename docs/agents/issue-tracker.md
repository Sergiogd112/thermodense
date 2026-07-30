# Issue tracker: GitHub

Issues, specifications, and roadmaps for this repository live as GitHub issues. Use the `gh` CLI for all operations and infer the repository from the current clone.

## General operations

- Create: `gh issue create --title "..." --body "..."`; use a body file for multiline content.
- Read: `gh issue view <number> --comments --json number,title,body,state,assignees,labels,url`.
- List: `gh issue list --state open --json number,title,body,assignees,labels,url` with suitable filters.
- Comment: `gh issue comment <number> --body "..."`.
- Label: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`.
- Assign or claim: `gh issue edit <number> --add-assignee "@me"`.
- Close: `gh issue close <number> --comment "..."`.

When a skill says to publish an issue, create a GitHub issue. When it says to fetch a ticket, read the issue and its comments.

## Wayfinding operations

Wayfinder uses GitHub's native sub-issue and dependency relationships so the map remains visually navigable. Obtain node or database IDs with `gh api` as needed; consult the current GitHub API schema if an endpoint has changed.

- A map is an issue labelled `wayfinder:map`.
- A decision ticket is a native sub-issue of its map and has exactly one of `wayfinder:research`, `wayfinder:prototype`, `wayfinder:grilling`, or `wayfinder:task`.
- Create all map and ticket issues first, then add sub-issue and dependency relationships in a second pass.
- Express ordering with native blocked-by relationships. Do not encode dependencies only in issue-body checklists.
- Claim a frontier ticket before working by assigning it to `@me`.
- The frontier is the map's open, unassigned sub-issues whose blocked-by dependencies are all closed.
- Resolve a ticket by posting its answer as a comment, closing it, and adding a linked one-line gist under the map's `Decisions so far` section.

Prefer the GitHub REST sub-issue and issue-dependency endpoints through `gh api`. If the installed GitHub API or account does not expose native relationships, record explicit `Parent map: <URL>` and `Blocked by: <URLs>` lines in ticket bodies as the documented fallback.

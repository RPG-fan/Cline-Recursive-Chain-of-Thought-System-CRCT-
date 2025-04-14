import argparse
import sys

def configure_cline_extension(args):
    from cline_utils.cli.commands import configure
    configure.main(args)

def initialize_dependencies(args):
    from cline_utils.cli.commands import init_deps
    init_deps.main(args)

def container_health_check(args):
    from cline_utils.cli.commands import check
    check.main(args)

def generate_project_template(args):
    from cline_utils.cli.commands import new
    new.main(args)

def update_crct(args):
    from cline_utils.cli.commands import update
    update.main(args)

def main():
    parser = argparse.ArgumentParser(
        description="CRCT Unified CLI - Container-First Automation Tools"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # configure
    parser_configure = subparsers.add_parser("configure", help="Configure Cline extension (copy core prompt, etc.)")
    parser_configure.set_defaults(func=configure_cline_extension)

    # init-deps
    parser_init_deps = subparsers.add_parser("init-deps", help="Initialize dependency trackers for all projects")
    parser_init_deps.set_defaults(func=initialize_dependencies)

    # check
    parser_check = subparsers.add_parser("check", help="Run container health check")
    parser_check.set_defaults(func=container_health_check)

    # new
    parser_new = subparsers.add_parser("new", help="Generate new project template")
    parser_new.set_defaults(func=generate_project_template)

    # update
    parser_update = subparsers.add_parser("update", help="Check for and apply CRCT updates")
    parser_update.set_defaults(func=update_crct)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

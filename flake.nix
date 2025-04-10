{
  description = "python shell flake";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          venvDir = ".venv";
          packages = [pkgs.pyright pkgs.black] ++ (with pkgs.python311Packages; [matplotlib numpy pip venvShellHook]);
        };
      }
    );
}

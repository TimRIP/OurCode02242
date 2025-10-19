{
  description = "JPAMB: Java Program Analysis Micro Benchmarks";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/ca77296380960cd497a765102eeb1356eb80fed0";
    flake-utils.url = "github:numtide/flake-utils";
    jvm2json.url = "github:kalhauge/jvm2json";
    jvm2json.inputs.nixpkgs.follows = "nixpkgs";
  };
  outputs =
    {
      nixpkgs,
      flake-utils,
      jvm2json,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells = {
          default = pkgs.mkShell {
            name = "jpamb";
            packages = with pkgs; [
              jdt-language-server
              jdk
              maven
              jvm2json.packages.${system}.default
              uv
            ];
          };
        };
      }
    );
}

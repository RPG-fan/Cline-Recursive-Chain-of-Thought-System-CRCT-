import type { Foo } from "./types";
import Bar, { baz as qux, type Quux } from "./bar";
import * as ns from "./ns.js";
import "./side-effect";
export * from "./reexp";
export * as ns2 from "./reexp2";
export { a, b as c } from "./reexp3";
export type { T1 } from "./types";

export interface Shape { area(): number }
export type Id = string | number;
export enum Color { Red, Green }
export const ARROW = (x: number): number => x * 2;
export const CONST_VAL = 42;
export function plain(a: Foo): Id { return a.id; }
export function overload(a: string): string;
export function overload(a: number): number;
export function overload(a: any): any { return a; }
export abstract class Base<T> implements Shape {
  abstract area(): number;
  protected helper(x: T): T { return x; }
  static make(): Base<number> { throw new Error(); }
  get name(): string { return "n"; }
  private field = 1;
}
export class Derived extends Base<number> {
  area(): number { return 1; }
  async load() { const m = await import("./lazy"); return m; }
}
export default class DefaultCls {}
declare module "foo" { export const x: number; }
namespace NS { export function inner() {} }
const local = () => 1;
function useAll() { return [ns, qux, Bar, local, NS]; }

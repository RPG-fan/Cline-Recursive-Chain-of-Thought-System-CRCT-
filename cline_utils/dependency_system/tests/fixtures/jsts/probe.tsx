import React from "react";
import { Btn } from "./btn";
export const Comp = ({ n }: { n: number }) => <Btn onClick={() => n}>x</Btn>;
export default function Page() { return <Comp n={1} />; }

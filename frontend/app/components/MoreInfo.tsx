import { Tooltip } from "react-tooltip";

const MoreInfo = function ({ message }: { message: string }) {
  const id = crypto.randomUUID();

  return (
    <div className="text-sm">
      <span
        data-tooltip-id={id}
        data-tooltip-content={message}
        className="material-symbols-outlined absolute top-1 right-1 text-[14px] opacity-0 group-hover:opacity-100 text-white cursor-help rounded-full transition"
      >
        help
      </span>
      <Tooltip
        id={id}
        place="top"
        variant="info"
        className="bg-primary"
        positionStrategy="fixed"
        offset={8}
      />
    </div>
  );
};
export default MoreInfo;
